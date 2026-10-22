"""
failure_cases.py - Tim cac truong hop robot DI SAI va ve hinh minh hoa.

    python failure_cases.py                          # huan luyen lai cac lan chay can thiet, do, luu, ve (~5 phut)
    python failure_cases.py --plot-only              # chi ve lai tu results/failure_cases.json
    python failure_cases.py --plot-only --lang en    # nhan tieng Anh, luu vao figures_en/

"Do chinh xac" trong experiments.py chi xet trang thai xuat phat (o trong bat ky, moi kien dang cho).
O day robot duoc dat vao MOI trang thai dat toi chua giao het hang (dang mang kien, da giao mot phan,
dong ho deadline da chay, tang tai dang bat...) roi chay policy tham lam va so voi Value Iteration.
Moi trang thai "di sai" (return thap hon toi uu) duoc xep vao 1 loai, theo thu tu uu tien:
  khong_xong : het max_steps ma chua giao het (robot lap vong)
  tre        : bi phat tre deadline trong khi lo trinh toi uu thi khong
  sai_luat   : co Pickup / Drop sai (-10)
  di_vong    : chi di nhieu buoc hon toi uu
Bon nhom lan chay (5 seed moi nhom):
  moi        : cau hinh da cai tien (ALGOS); seed 0 dung bang Q da luu trong results/, seed khac huan luyen lai
  cu         : cau hinh cu (BASELINE), cung so episode
  sarsa_x2   : SARSA cau hinh moi nhung gap doi so episode (chi S3, S4 - hai map SARSA con sai)
  sarsa_a002 : SARSA cau hinh moi nhung alpha giam toi 0.02 thay vi 0.1 (chi S3, S4)
Huan luyen tat dinh theo seed nen cac lan chay lai trung voi lan chay trong experiments.py.
Ket qua: results/failure_cases.json (tong hop) va results/failure_states.csv (moi trang thai di sai: ma trang thai
s = env.encode(...), o robot, trang thai tung kien, tau, boost, hanh dong tham lam / toi uu, so diem bi mat).
"""
import argparse
import csv
import glob
import json
import os
import time
from collections import Counter
from multiprocessing import Pool

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Rectangle

import plots as P
from agents import load_q, run_greedy
from delivery_env import CARRYING, DELIVERED, DROP, PICKUP, MultiDeliveryEnv, load_config
from experiments import ALGOS, build_run, route_summary
from planner import value_iteration

OUT = os.path.join("results", "failure_cases.json")
STATES = os.path.join("results", "failure_states.csv")  # tung trang thai di sai kem ma trang thai
SEEDS = [0, 1, 2, 3, 4]
KINDS = ["khong_xong", "tre", "sai_luat", "di_vong"]
ERR = "#d03b3b"  # mau bao loi: vong lap cua robot
plt = P.plt


# ======================================================================
# Chay policy tham lam tu nhieu trang thai cung luc
# ======================================================================
def fields(env, idx):
    """Mang chi so trang thai -> (o, ma kien, tau, boost): lam nguoc encode() bang divmod."""
    idx, boost = np.divmod(idx, env.n_boost)
    idx, tau = np.divmod(idx, env.n_tau)
    pos, code = np.divmod(idx, env.n_codes)
    return pos, code, tau, boost


def counts(env, code):
    """So kien da giao / dang mang tu ma kien (so co so 3)."""
    digits = [(code // 3 ** i) % 3 for i in range(env.N)]
    return sum(d == DELIVERED for d in digits), sum(d == CARRYING for d in digits)


def rollout(env, Q, starts):
    """Chay policy tham lam (eps = 0) tu moi trang thai trong starts, ca mang mot luc qua bang NS/R/D.
    Tra ve return (khong chiet khau), so buoc, giao het?, so lan Pickup/Drop sai, bi phat tre?"""
    NS, R, D = env.tables()
    s = np.asarray(starts, dtype=np.int64).copy()
    n = len(s)
    out = {"ret": np.zeros(n), "steps": np.zeros(n, dtype=np.int64), "done": np.zeros(n, dtype=bool),
           "illegal": np.zeros(n, dtype=np.int64), "late": np.zeros(n, dtype=bool)}
    for _ in range(env.max_steps):
        run = np.flatnonzero(~out["done"])
        if not len(run):
            break
        cur = s[run]
        a = Q[cur].argmax(axis=1)  # argmax lay chi so dau, giong run_greedy
        nxt = NS[cur, a]
        _, c0, t0, _ = fields(env, cur)
        _, c1, t1, _ = fields(env, nxt)
        out["ret"][run] += R[cur, a]
        out["steps"][run] += 1
        out["illegal"][run] += ((a == PICKUP) | (a == DROP)) & (c0 == c1)  # ma kien khong doi = lam sai
        if env.dl_idx is not None:  # dong ho vua cham T_d ma kien deadline chua giao -> bi phat tre
            open_ = (c1 // 3 ** env.dl_idx) % 3 != DELIVERED
            out["late"][run] |= open_ & (t0 == env.deadline - 1) & (t1 == env.deadline)
        out["done"][run] = D[cur, a]
        s[run] = nxt
    return out


def path(env, Q, s):
    """Quy dao tham lam (chi so trang thai) tu s. Dung khi giao het, het max_steps hoac gap lai mot
    trang thai cu (policy tat dinh -> robot se lap vong mai). Tra ve (quy dao, vi tri bat dau vong lap)."""
    NS, _, D = env.tables()
    s = int(s)
    traj, seen = [s], {s: 0}
    for _ in range(env.max_steps):
        a = int(np.argmax(Q[s]))
        done, s = bool(D[s, a]), int(NS[s, a])
        traj.append(s)
        if done:
            return traj, None
        if s in seen:
            return traj, seen[s]
        seen[s] = len(traj) - 1
    return traj, None


# ======================================================================
# Do loi cua 1 bang Q
# ======================================================================
_REF = {}


def reference(cfg):
    """(env, Q* tu Value Iteration, trang thai dat toi chua giao het, ket qua tham lam cua Q*). Nho theo map."""
    if cfg["name"] not in _REF:
        env = MultiDeliveryEnv(cfg, "depot")
        Qs, _ = value_iteration(env, gamma=cfg["train"]["gamma"])
        S = env.start_states("exploring")
        _REF[cfg["name"]] = (env, Qs, S, rollout(env, Qs, S))
    return _REF[cfg["name"]]


def q_row(Q, Qs, s):
    """Hang Q hoc duoc va Q* tai trang thai s + hanh dong tham lam / cac hanh dong toi uu."""
    return {"q": [round(float(v), 2) for v in Q[s]], "q_star": [round(float(v), 2) for v in Qs[s]],
            "action": int(np.argmax(Q[s])), "best": [int(k) for k in np.flatnonzero(Qs[s] >= Qs[s].max() - 1e-6)]}


def decisions(env, Q, wrong, starts):
    """Lan theo quy dao tham lam tu moi trang thai trong starts toi trang thai DAU TIEN ma hanh dong tham lam
    khong toi uu theo Q* (wrong: mang bool theo chi so trang thai) = noi robot re sai.
    Tra ve (chi so trang thai quyet dinh sai hoac -1, so buoc di toi do)."""
    NS, _, D = env.tables()
    cur = np.asarray(starts, dtype=np.int64).copy()
    dec, dist = np.full(len(cur), -1, dtype=np.int64), np.zeros(len(cur), dtype=np.int64)
    live = np.ones(len(cur), dtype=bool)
    for _ in range(env.max_steps):
        hit = live & wrong[cur]
        dec[hit] = cur[hit]
        live &= ~hit
        if not live.any():
            break
        act = Q[cur].argmax(axis=1)
        live &= ~D[cur, act]
        cur = np.where(live, NS[cur, act], cur)
        dist += live
    return dec, dist


def example(env, Q, Qs, s, gap, wrong):
    traj, cyc = path(env, Q, s)
    me, opt = rollout(env, Q, [s]), rollout(env, Qs, [s])
    dec, dist = decisions(env, Q, wrong, [s])
    return {"state": int(s), "gap": gap, "learned": traj, "cycle": cyc, "optimal": path(env, Qs, s)[0],
            "learned_return": float(me["ret"][0]), "learned_steps": int(me["steps"][0]),
            "optimal_return": float(opt["ret"][0]), "optimal_steps": int(opt["steps"][0]), **q_row(Q, Qs, s),
            "decision": int(dec[0]), "decision_step": int(dist[0]),
            "decision_q": q_row(Q, Qs, int(dec[0])) if dec[0] >= 0 else None}


def analyse(cfg, Q, all_states=True):
    """Do loi cua bang Q tren moi trang thai dat toi chua giao het.
    all_states=False: chi ghi chi tiet (rows) cac trang thai xuat phat di sai, de file CSV khong qua lon."""
    env, Qs, S, opt = reference(cfg)
    me = rollout(env, Q, S)
    sub = me["done"] & (me["ret"] < opt["ret"] - 1e-9)
    bad = sub | ~me["done"]
    late = sub & me["late"] & ~opt["late"]
    illegal = sub & ~late & (me["illegal"] > 0)
    kinds = {"khong_xong": ~me["done"], "tre": late, "sai_luat": illegal, "di_vong": sub & ~late & ~illegal}
    a = Q[S].argmax(axis=1)
    wrong_action = Qs[S, a] < Qs[S].max(axis=1) - 1e-6  # giong pol_acc trong convergence.py
    wrong = np.zeros(env.n_states, dtype=bool)
    wrong[S] = wrong_action
    bad_idx = np.flatnonzero(bad)
    dec, dist = decisions(env, Q, wrong, S[bad_idx])  # trang thai re sai cua tung trang thai di sai
    dec_count = Counter(int(d) for d in dec)

    pos, code, _, boost = fields(env, S[bad])
    phases = Counter()
    for c, b in zip(code, boost):
        d, k = counts(env, c)
        phases[f"{d}|{k}|{b}"] += 1
    gap = opt["ret"] - me["ret"]
    ex = {}
    if sub.any():  # trang thai giao het nhung thiet nhieu nhat
        i = np.flatnonzero(sub)[np.argmax(gap[sub])]
        ex["worst"] = example(env, Q, Qs, S[i], float(gap[i]), wrong)
    if (~me["done"]).any():
        i = np.flatnonzero(~me["done"])[0]
        ex["loop"] = example(env, Q, Qs, S[i], None, wrong)

    cells = env.start_states("random_cell")
    mc, oc = rollout(env, Q, cells), rollout(env, Qs, cells)
    ev = MultiDeliveryEnv(cfg, "depot")
    res = run_greedy(ev, Q)
    s0 = ev.start_states("depot")[0]
    traj, cyc = path(env, Q, s0)

    # tung trang thai di sai: ma trang thai, loai loi, trang thai re sai + hanh dong o do -> results/failure_states.csv
    is_start = np.isin(S, cells)
    kind_of = np.select([kinds[k] for k in KINDS], KINDS, "")
    rows = []
    for j, i in enumerate(bad_idx):
        if all_states or is_start[i]:
            d = int(dec[j])
            rows.append([int(S[i]), str(kind_of[i]), float(me["ret"][i]), float(opt["ret"][i]), int(me["steps"][i]),
                         int(opt["steps"][i]), d, int(dist[j]), int(np.argmax(Q[d])) if d >= 0 else -1,
                         [int(k) for k in np.flatnonzero(Qs[d] >= Qs[d].max() - 1e-6)] if d >= 0 else [],
                         bool(is_start[i])])
    # cac trang thai re sai pho bien nhat: (ma trang thai, so trang thai di sai dan toi day, Q, Q*)
    top = [dict(state=d, n=n, **q_row(Q, Qs, d)) for d, n in dec_count.most_common(12) if d >= 0]
    return {
        "n_states": int(len(S)), "n_bad": int(bad.sum()),
        "kinds": {k: int(v.sum()) for k, v in kinds.items()},
        "wrong_action": int(wrong_action.sum()),
        "worst_gap": float(gap[sub].max()) if sub.any() else 0.0,
        "mean_gap": float(gap[sub].mean()) if sub.any() else 0.0,
        "phases": dict(phases),
        "cells": np.bincount(pos, minlength=env.n_pos).reshape(env.H, env.W).tolist(),
        "start_cells_bad": int((~mc["done"] | (mc["ret"] < oc["ret"] - 1e-9)).sum()),
        "n_start_cells": int(len(cells)),
        "from_S": dict(route_summary(res), illegal=sum(e.startswith(("pickup sai", "drop sai"))
                                                       for _, e in res["events"]),
                       delivered=sum(e.startswith("giao kien") for _, e in res["events"]),
                       path=traj, cycle=cyc,
                       loop=None if cyc is None else dict(state=traj[cyc], **q_row(Q, Qs, traj[cyc]))),
        "examples": ex,
        "n_decisions": len([d for d in dec_count if d >= 0]),
        "decisions": top,
        "rows": rows,
    }


def run_one(job):
    """1 lan chay (chay trong process con): lay / huan luyen bang Q roi do loi."""
    cfg, group, key, seed, scale = job
    t0 = time.time()
    if group == "moi" and seed == 0:  # bang Q experiments.py da luu = bang dung trong bao cao
        Q = load_q(os.path.join("results", f"Q_{cfg['name']}_{key}.npz"))
    else:
        fn, env, ev, kw = build_run(cfg, key, seed, scale)
        if group == "sarsa_a002":
            kw["alpha_end"] = 0.02  # buoc hoc cuoi nho hon -> lay trung binh nhieu muc tieu nhieu hon
        Q, _ = fn(env, ev, **kw)
    out = analyse(cfg, Q, all_states=group != "cu")  # cau hinh cu sai hang chuc nghin trang thai
    out.update(name=cfg["name"], group=group, algo=key.replace("_cu", ""), seed=seed, scale=scale,
               time=time.time() - t0)
    return out


def run_all(cfgs, workers=None):
    jobs = []
    for c in cfgs:
        for key in ALGOS:
            jobs += [(c, "moi", key, s, 1.0) for s in SEEDS] + [(c, "cu", f"{key}_cu", s, 1.0) for s in SEEDS]
        if c["level"] >= 3:
            jobs += [(c, "sarsa_x2", "sarsa", s, 2.0) for s in SEEDS] + [(c, "sarsa_a002", "sarsa", s, 1.0) for s in SEEDS]
    jobs.sort(key=lambda j: -j[0]["train"]["episodes"] * j[4])  # job dai chay truoc
    runs, t0 = [], time.time()
    print(f"Do loi tren moi trang thai dat toi: {len(jobs)} lan chay")
    with Pool(workers) as pool:
        for i, r in enumerate(pool.imap_unordered(run_one, jobs), 1):
            print(f"  [{i}/{len(jobs)}] {r['name']} {r['group']} {r['algo']} seed={r['seed']}: "
                  f"sai {r['n_bad']}/{r['n_states']} | tu S return {r['from_S']['return']:.0f} ({r['time']:.0f}s)",
                  flush=True)
            runs.append(r)
    runs.sort(key=lambda r: (r["name"], r["group"], r["algo"], r["seed"]))
    maps = {}
    for c in cfgs:
        env, Qs, S, _ = reference(c)
        maps[c["name"]] = {"n_states": int(len(S)), "max_steps": env.max_steps,
                           "optimal": route_summary(run_greedy(MultiDeliveryEnv(c, "depot"), Qs))}
    write_states(cfgs, runs)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({"seeds": SEEDS, "maps": maps, "runs": runs}, f)
    print(f"Xong sau {time.time() - t0:.0f}s ->", OUT)


ACTIONS = ["Up", "Down", "Left", "Right", "Pickup", "Drop"]
STATUS = ["waiting", "carried", "delivered"]
# ten tieng Anh trong file CSV (bao cao tieng Anh tro toi file nay)
GROUP_EN = {"moi": "new", "cu": "old", "sarsa_x2": "new_2x_episodes", "sarsa_a002": "new_final_alpha_0.02"}
KIND_EN = {"khong_xong": "never_finishes", "tre": "late", "sai_luat": "illegal_action", "di_vong": "longer_route"}


def state_txt(env, s):
    """Ma trang thai -> '(row, col, s1..sN, tau, b)' (cung ky hieu voi bao cao)."""
    r, c, status, tau, boost = env.decode(int(s))
    return "(" + ", ".join(str(v) for v in (r, c, *status, tau, boost)) + ")"


def write_states(cfgs, runs):
    """Moi trang thai di sai -> 1 dong CSV: ma trang thai (= env.encode, chi so hang cua bang Q) va giai ma
    (o, trang thai tung kien, tau, boost), cung trang thai RE SAI tren quy dao tu do (ma trang thai, giai ma,
    hanh dong robot chon / hanh dong toi uu). Cau hinh cu chi ghi cac trang thai xuat phat (o trong, chua lay kien)."""
    envs = {c["name"]: MultiDeliveryEnv(c, "depot") for c in cfgs}
    with open(STATES, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["map", "settings", "algorithm", "seed", "state_code", "state", "row", "col", "parcels", "tau",
                    "boost", "start_state", "mistake", "learned_return", "optimal_return", "lost_points",
                    "learned_steps", "optimal_steps", "wrong_decision_code", "wrong_decision_state",
                    "steps_to_wrong_decision", "action_taken", "optimal_actions"])
        n = 0
        for r in runs:
            env = envs[r["name"]]
            for s, kind, ret, oret, steps, osteps, d, dist, act, best, start in r.pop("rows"):
                row, col, status, tau, boost = env.decode(s)
                parcels = " ".join(f"{p['name']}={STATUS[x]}" for p, x in zip(env.parcels, status))
                w.writerow([P.short(r["name"]), GROUP_EN[r["group"]], P.ALGO_LABEL[r["algo"]], r["seed"], s,
                            state_txt(env, s), row, col, parcels, tau, boost, int(start), KIND_EN[kind], f"{ret:g}",
                            f"{oret:g}", f"{oret - ret:g}", steps, osteps, d if d >= 0 else "",
                            state_txt(env, d) if d >= 0 else "", dist if d >= 0 else "",
                            ACTIONS[act] if d >= 0 else "", "/".join(ACTIONS[k] for k in best)])
                n += 1
    print(f"{n} trang thai di sai ->", STATES)


# ======================================================================
# Ve hinh
# ======================================================================
def pick(data, name, group, algo):
    return [r for r in data["runs"] if r["name"] == name and r["group"] == group and r["algo"] == algo]


def rate(runs):
    """% trang thai di sai cua tung seed."""
    return np.array([100 * r["n_bad"] / r["n_states"] for r in runs])


def plot_rates(data):
    """% trang thai dat toi ma policy tham lam di sai: cau hinh cu (xam), moi (mau);
    rieng SARSA them 2 cot rong: gap doi episode (gach cheo) va alpha cuoi 0.02 (cham)."""
    names = list(data["maps"])
    fig, axes = plt.subplots(1, len(P.ALGO_COLOR), figsize=(13, 4.1), sharey=True)
    x = np.arange(len(names))
    for ax, key in zip(axes, P.ALGO_COLOR):
        bars = [("cu", P.OLD_COLOR, None), ("moi", P.ALGO_COLOR[key], None)]
        if key == "sarsa":
            bars += [("sarsa_x2", "white", "////"), ("sarsa_a002", "white", "....")]
        w = 0.84 / len(bars)
        for b, (group, fc, hatch) in enumerate(bars):
            dx = (b - (len(bars) - 1) / 2) * w
            for i, n in enumerate(names):
                runs = pick(data, n, group, key)
                if not runs:
                    continue
                v = rate(runs)
                ax.bar(x[i] + dx, v.mean(), w * 0.92, color=fc, ec=P.ALGO_COLOR[key] if hatch else fc,
                       hatch=hatch, lw=1)
                ax.scatter(np.full(len(v), x[i] + dx) + np.linspace(-0.05, 0.05, len(v)), v,
                           s=9, color=P.INK, zorder=5)
                m = v.mean()
                ax.text(x[i] + dx, max(m, v.max()) + 3, f"{m:.1f}" if 0 < m < 10 else f"{m:.0f}",
                        ha="center", fontsize=8, color=P.INK)
        ax.set_xticks(x, [P.short(n) for n in names])
        ax.set_ylim(0, 114)
        ax.set_title(P.ALGO_LABEL[key], loc="left", fontsize=10.5, color=P.INK)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel(P.L("% trạng thái đạt tới mà policy\nđi sai (kém tối ưu)",
                           "% reachable states where the\ngreedy policy goes wrong"))
    handles = [Rectangle((0, 0), 1, 1, fc=P.OLD_COLOR, label=P.L("Cấu hình cũ", "Old settings"))]
    handles += [Rectangle((0, 0), 1, 1, fc=P.ALGO_COLOR[k], label=P.L(f"{P.ALGO_LABEL[k]}, cấu hình mới",
                                                                   f"{P.ALGO_LABEL[k]}, new settings"))
                for k in P.ALGO_COLOR]
    handles.append(Rectangle((0, 0), 1, 1, fc="white", ec=P.ALGO_COLOR["sarsa"], hatch="////",
                             label=P.L("SARSA mới, gấp đôi episode", "SARSA new, 2× episodes")))
    handles.append(Rectangle((0, 0), 1, 1, fc="white", ec=P.ALGO_COLOR["sarsa"], hatch="....",
                             label=P.L("SARSA mới, α cuối 0,02", "SARSA new, final α 0.02")))
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.06), fontsize=9)
    fig.suptitle(P.L("Robot đặt vào MỌI trạng thái đạt tới: % trạng thái đi sai (cột = trung bình, chấm = từng seed)",
                     "Robot started in EVERY reachable state: % of states where it goes wrong (bar = mean, dots = seeds)"),
                 fontsize=12, fontweight="bold", color=P.INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    P.save(fig, "failure_rate.png")


def draw_path(ax, env, traj, cycle=None, decision=None):
    """Ve quy dao (chi so trang thai): net to mau theo kien dang mang, so = lan lay / giao,
    vien do = cac o robot lap mai (cycle = vi tri bat dau vong lap trong traj),
    vong tron do = o cua trang thai re sai (decision = ma trang thai)."""
    st = [env.decode(s) for s in traj]
    P.draw_grid(ax, env, st[0][2], st[0][4])
    phase = np.cumsum([0] + [a[2] != b[2] for a, b in zip(st[:-1], st[1:])])
    off = lambda k: (k - phase[-1] / 2) * 0.07  # lech nhe moi pha de doan di lai khong chong nhau
    for t, (a, b) in enumerate(zip(st[:-1], st[1:])):
        color, _ = P._carry_color(env, a[2])
        d = off(phase[t])
        ax.plot([a[1] + d, b[1] + d], [a[0] + d, b[0] + d], color=color, lw=2.5, solid_capstyle="round", zorder=5)
    color, _ = P._carry_color(env, st[0][2])
    ax.add_patch(Circle((st[0][1], st[0][0]), 0.17, fc=color, ec=P.SURF, lw=2, zorder=6))
    k = 0
    for a, b in zip(st[:-1], st[1:]):
        if a[2] != b[2]:  # vua lay hoac giao mot kien
            k += 1
            ax.add_patch(Circle((b[1] + 0.3, b[0] - 0.3), 0.17, fc="white", ec=P.INK, lw=1, zorder=7))
            ax.text(b[1] + 0.3, b[0] - 0.3, str(k), ha="center", va="center", fontsize=7.5, color=P.INK, zorder=8)
    if cycle is not None:
        for r, c in {(s[0], s[1]) for s in st[cycle:]}:
            ax.add_patch(Rectangle((c - 0.44, r - 0.44), 0.88, 0.88, fc="none", ec=ERR, lw=2.2, zorder=9))
    if decision is not None and decision >= 0:
        r, c = env.decode(int(decision))[:2]
        ax.add_patch(Circle((c, r), 0.46, fc="none", ec=ERR, lw=2.6, zorder=9))


def path_legend(fig, loop=False, decision=False):
    h = [Line2D([], [], color=P.EMPTY_PATH, lw=2.5, label=P.L("tay không", "empty-handed"))]
    h += [Line2D([], [], color=c, lw=2.5, label=P.L(f"mang {n}", f"carrying {n}")) for n, c in P.PARCEL_COLOR.items()]
    h += [Line2D([], [], color=P.INK, lw=2.5, label=P.L("mang 2 kiện", "carrying 2")),
          Line2D([], [], marker="o", ls="", mfc="white", mec=P.INK, ms=9,
                 label=P.L("lần lấy / giao thứ k", "k-th pickup / drop"))]
    if loop:
        h.append(Rectangle((0, 0), 1, 1, fc="none", ec=ERR, lw=2,
                           label=P.L("ô robot lặp vòng mãi", "cells the robot loops on")))
    if decision:
        h.append(Line2D([], [], marker="o", ls="", mfc="none", mec=ERR, mew=2.2, ms=12,
                        label=P.L("chỗ rẽ sai", "wrong decision")))
    fig.legend(handles=h, loc="lower center", ncol=len(h), bbox_to_anchor=(0.5, -0.01), fontsize=9)


def outcome(r, env):
    """Mo ta ket qua tu o S cua 1 lan chay."""
    f = r["from_S"]
    if not f["success"]:
        cyc = f["path"][f["cycle"]:] if f["cycle"] is not None else []
        cells = len({env.decode(s)[:2] for s in cyc})
        return P.L(f"giao {f['delivered']}/{env.N} · return {f['return']:.0f}\n"
                   f"lặp vòng trên {cells} ô, trạng thái s = {f['loop']['state']}",
                   f"delivers {f['delivered']}/{env.N} · return {f['return']:.0f}\n"
                   f"loops on {cells} cell{'s' if cells > 1 else ''}, state s = {f['loop']['state']}")
    txt = P.L(f"thứ tự {'→'.join(f['order'].split('-'))} · {f['steps']} bước · return {f['return']:.0f}",
              f"order {'→'.join(f['order'].split('-'))} · {f['steps']} steps · return {f['return']:.0f}")
    if f["late"]:
        txt += P.L(" · TRỄ hạn", " · LATE")
    return txt


def plot_old_routes(data, cfgs):
    """Moi map: lan chay cau hinh cu te nhat khi xuat phat o S (khong giao het > tre > return thap)."""
    fig, axes = plt.subplots(1, len(cfgs), figsize=(4.0 * len(cfgs), 5.0))
    for ax, cfg in zip(axes, cfgs):
        env = MultiDeliveryEnv(cfg, "depot")
        runs = [r for r in data["runs"] if r["name"] == cfg["name"] and r["group"] == "cu"]
        r = min(runs, key=lambda r: (r["from_S"]["success"], not r["from_S"]["late"], r["from_S"]["return"]))
        f = r["from_S"]
        opt = data["maps"][cfg["name"]]["optimal"]
        draw_path(ax, env, f["path"], f["cycle"])  # khong giao het: quy dao den khi gap lai trang thai cu
        ax.set_title(P.L(f"{P.short(cfg['name'])} · {P.ALGO_LABEL[r['algo']]} (cũ), seed {r['seed']}\n",
                         f"{P.short(cfg['name'])} · {P.ALGO_LABEL[r['algo']]} (old), seed {r['seed']}\n")
                     + outcome(r, env) + P.L(f"\ntối ưu: {opt['steps']} bước, return {opt['return']:.0f}",
                                             f"\noptimal: {opt['steps']} steps, return {opt['return']:.0f}"),
                     fontsize=9, color=P.INK)
    path_legend(fig, loop=True)
    fig.suptitle(P.L("Cấu hình cũ: lần chạy tệ nhất trên mỗi map khi xuất phát ở S",
                     "Old settings: the worst run on each map when starting at S"),
                 fontsize=12, fontweight="bold", color=P.INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.05, 1, 0.92))
    P.save(fig, "failure_old_routes.png")


def plot_sarsa_case(data, cfg):
    """SARSA cau hinh moi tren S4: trang thai thiet nhieu nhat (seed 0) - hoc duoc vs toi uu,
    va ban do so trang thai di sai theo vi tri robot (cong 5 seed)."""
    env = MultiDeliveryEnv(cfg, "depot")
    runs = pick(data, cfg["name"], "moi", "sarsa")
    ex = runs[0]["examples"]["worst"]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 5.2))
    st = env.decode(ex["state"])
    held = "+".join(env.parcels[i]["name"] for i, s in enumerate(st[2]) if s == CARRYING) or "–"
    for ax, key, lab in [(axes[0], "learned", P.L("SARSA học được", "SARSA (learned)")),
                         (axes[1], "optimal", P.L("Tối ưu (Value Iteration)", "Optimal (Value Iteration)"))]:
        draw_path(ax, env, ex[key], decision=ex["decision"])
        n, ret = ex[f"{key}_steps"], ex[f"{key}_return"]
        ax.set_title(f"{lab}\n" + P.L(f"{n} bước · return {ret:.0f}", f"{n} steps · return {ret:.0f}"),
                     fontsize=10, color=P.INK)
    # ban do: robot dang o dau khi policy di sai (cong 5 seed)
    cnt = np.sum([np.array(r["cells"]) for r in runs], axis=0)
    P.draw_grid(axes[2], env)
    vmax = cnt.max()
    for r in range(env.H):
        for c in range(env.W):
            if cnt[r, c] > 0:
                axes[2].add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, fc=P.HEAT_CMAP(cnt[r, c] / vmax), ec="white",
                                            lw=1, zorder=3))
                axes[2].text(c, r, f"{cnt[r, c]:.0f}", ha="center", va="center", fontsize=8.5, zorder=4,
                             color="white" if cnt[r, c] / vmax > P.HEAT_DARK else P.INK)
    tot = sum(r["n_bad"] for r in runs)
    axes[2].set_title(P.L(f"Vị trí robot ở các trạng thái đi sai\n(cộng {len(runs)} seed: {tot:,} trạng thái"
                          f" = {rate(runs).mean():.1f}% mỗi seed)",
                          f"Where the robot is in the wrong states\n({len(runs)} seeds together: {tot:,} states"
                          f" = {rate(runs).mean():.1f}% per seed)"),
                      fontsize=10, color=P.INK)
    boost = [P.L("chưa bật", "off"), P.L("ĐANG BẬT", "ON"), P.L("đã hết", "used")][st[4]]
    fig.suptitle(P.L(f"S4 · SARSA (cấu hình mới, seed 0), trạng thái s = {ex['state']}: robot ở ({st[0]},{st[1]}), "
                     f"đang mang {held}, tăng tải {boost}, τ = {st[3]} → thiệt {ex['gap']:.0f} điểm",
                     f"S4 · SARSA (new settings, seed 0), state s = {ex['state']}: robot at ({st[0]},{st[1]}), "
                     f"carrying {held}, booster {boost}, τ = {st[3]} → loses {ex['gap']:.0f} points"),
                 fontsize=12, fontweight="bold", color=P.INK, x=0.01, ha="left")
    dq = ex["decision_q"]
    dr, dc = env.decode(ex["decision"])[:2]
    axes[0].text(0.5, -0.04, P.L(f"rẽ sai ở trạng thái {ex['decision']} = {state_txt(env, ex['decision'])}, ô ({dr},{dc}):\n"
                                 f"SARSA chọn {ACTIONS[dq['action']]}, tối ưu là {'/'.join(ACTIONS[k] for k in dq['best'])}",
                                 f"wrong decision at state {ex['decision']} = {state_txt(env, ex['decision'])}, cell ({dr},{dc}):\n"
                                 f"SARSA takes {ACTIONS[dq['action']]}, optimal is {'/'.join(ACTIONS[k] for k in dq['best'])}"),
                 transform=axes[0].transAxes, ha="center", va="top", fontsize=9, color=ERR)
    path_legend(fig, decision=True)
    fig.tight_layout(rect=(0, 0.09, 1, 0.95))  # chua cho dong chu do duoi o trai + chu thich
    P.save(fig, "failure_sarsa_S4.png")


def report(data):
    print("\n% trang thai dat toi di sai (trung bình 5 seed) | loai loi (tong 5 seed) | tu S:")
    for name, m in data["maps"].items():
        print(f"{name}: {m['n_states']} trang thai, toi uu tu S {m['optimal']}")
        for group in ("cu", "moi", "sarsa_x2", "sarsa_a002"):
            for key in P.ALGO_COLOR:
                runs = pick(data, name, group, key)
                if not runs:
                    continue
                kinds = {k: sum(r["kinds"][k] for r in runs) for k in KINDS}
                fs = [f"{r['from_S']['return']:.0f}{'' if r['from_S']['success'] else '!'}" for r in runs]
                print(f"  {group:8s} {key:5s} {rate(runs).mean():6.2f}% | " +
                      " ".join(f"{k}={v}" for k, v in kinds.items()) +
                      f" | max thiet {max(r['worst_gap'] for r in runs):.0f} | tu S: {' '.join(fs)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", choices=["vi", "en"], default="vi", help="ngon ngu nhan tren hinh")
    ap.add_argument("--plot-only", action="store_true", help="khong huan luyen lai, ve tu results/failure_cases.json")
    ap.add_argument("--workers", type=int, default=None, help="so process song song (mac dinh: moi nhan CPU)")
    args = ap.parse_args()
    cfgs = [load_config(p) for p in sorted(glob.glob("configs/*.json"))]
    if not (args.plot_only and os.path.exists(OUT)):
        run_all(cfgs, args.workers)
    P.LANG = args.lang
    if args.lang == "en":
        P.FIG = "figures_en"
        P.ALGO_LABEL.update(P.ALGO_LABEL_EN)
        P.TITLE.update(P.TITLE_EN)
    with open(OUT, encoding="utf-8") as f:
        data = json.load(f)
    report(data)
    plot_rates(data)
    plot_old_routes(data, cfgs)
    plot_sarsa_case(data, next(c for c in cfgs if c["level"] == 4))


if __name__ == "__main__":
    main()
