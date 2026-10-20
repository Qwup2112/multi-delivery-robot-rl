"""
convergence.py - Ve do hoi tu cua bang Q ve Q* (Q toi uu tinh bang Value Iteration).

    python convergence.py                         # huan luyen lai (cung seed, cung tham so voi experiments.py) roi ve
    python convergence.py --plot-only             # chi ve lai tu results/convergence.npz
    python convergence.py --plot-only --lang en   # nhan tieng Anh, luu vao figures_en/

Moi lan danh gia (moi eval_every episode) ghi lai:
  q_err     = trung binh |Q(s,a) - Q*(s,a)| tren moi trang thai dat toi (chua giao het hang)
  pol_acc   = ti le trang thai dat toi ma hanh dong tham lam argmax_a Q(s,a) la mot hanh dong toi uu
  pol_route = cung ti le do nhung chi tren cac trang thai thuoc lo trinh toi uu tu moi o xuat phat
              (phan lon trang thai dat toi chi gap khi robot di sai luc kham pha, nen pol_acc thap hon)
  q_start   = hang Q(S, .) tai o xuat phat S
Callback khong dung bo sinh so ngau nhien nen cac lan chay trung voi lan chay trong bao cao.
"""
import argparse
import glob
import os
import time
from multiprocessing import Pool

import numpy as np
from matplotlib.lines import Line2D

import plots as P
from delivery_env import DELIVERED, DOWN, PICKUP, RIGHT, MultiDeliveryEnv, load_config
from experiments import ALGOS, build_run
from planner import value_iteration

kilo = P.kilo

OUT = os.path.join("results", "convergence.npz")
SEEDS = [0, 1, 2, 3, 4]
ERR_FLOOR = 1e-4  # day truc log cua sai so |Q - Q*|
plt = P.plt


def reference(cfg):
    """Q* tu Value Iteration, cac trang thai dat toi (chua giao het), mat na hanh dong toi uu
    va vi tri (trong idx) cua cac trang thai nam tren lo trinh toi uu tu moi o xuat phat."""
    env = MultiDeliveryEnv(cfg)
    Qs, _ = value_iteration(env, gamma=cfg["train"]["gamma"])
    idx = np.array(sorted(env.encode(s) for s in env.reachable_states()
                          if not all(x == DELIVERED for x in s[2])))
    best = Qs[idx].max(axis=1, keepdims=True)
    pos = {s: i for i, s in enumerate(idx)}
    route = set()
    for r, c in env.free_cells:  # moi o xuat phat luc huan luyen, moi kien dang cho
        state = (r, c, (0,) * env.N, 0, 0)
        for _ in range(cfg["max_steps"]):
            s = env.encode(state)
            route.add(pos[s])
            state, _, done, _ = env.transition(state, int(np.argmax(Qs[s])))
            if done:
                break
    return Qs, idx, Qs[idx] >= best - 1e-6, np.array(sorted(route))


def run_one(job):
    """1 lan huan luyen co theo doi hoi tu (chay trong process con)."""
    cfg, key, seed = job
    Qs, idx, opt, route = reference(cfg)
    fn, env, ev, kw = build_run(cfg, key, seed)
    s0, _ = MultiDeliveryEnv(cfg, "depot").reset()  # trang thai xuat phat: o S, chua lay kien nao
    rows = np.arange(len(idx))
    rec = {"ep": [], "q_err": [], "pol_acc": [], "pol_route": [], "q_start": []}

    def on_eval(ep, Q):
        q = Q[idx]
        ok = opt[rows, q.argmax(axis=1)]  # argmax lay chi so dau, giong run_greedy
        rec["ep"].append(ep)
        rec["q_err"].append(np.abs(q - Qs[idx]).mean())
        rec["pol_acc"].append(ok.mean())
        rec["pol_route"].append(ok[route].mean())
        rec["q_start"].append(Q[s0].copy())

    t0 = time.time()
    fn(env, ev, on_eval=on_eval, **kw)
    return cfg["name"], key, seed, {k: np.array(v) for k, v in rec.items()}, time.time() - t0


def run_all(cfgs, workers=None):
    jobs = [(c, k, s) for c in cfgs for k in ALGOS for s in SEEDS]
    jobs.sort(key=lambda j: -j[0]["train"]["episodes"] * (2 if j[1] == "mc" else 1))  # job dai chay truoc
    arrays, t0 = {}, time.time()
    print(f"Huan luyen lai {len(jobs)} lan chay de do hoi tu:")
    with Pool(workers) as pool:
        for i, (name, key, seed, rec, dt) in enumerate(pool.imap_unordered(run_one, jobs), 1):
            print(f"  [{i}/{len(jobs)}] {name} {key} seed={seed} ({dt:.0f}s)", flush=True)
            for k, v in rec.items():
                arrays[f"{name}|{key}|{seed}|{k}"] = v
    for c in cfgs:  # Q*(S, .) de ve duong tham chieu
        Qs = reference(c)[0]
        s0, _ = MultiDeliveryEnv(c, "depot").reset()
        arrays[f"{c['name']}|qstar_start"] = Qs[s0]
    np.savez_compressed(OUT, **arrays)
    print(f"Xong sau {time.time() - t0:.0f}s ->", OUT)


def stack(data, name, key, metric):
    """(episode, mang [seed x lan danh gia]) cua 1 chi so."""
    runs = [data[f"{name}|{key}|{s}|{metric}"] for s in SEEDS]
    return data[f"{name}|{key}|{SEEDS[0]}|ep"], np.array(runs)


def plot_convergence(cfgs, data):
    """Hang 1: sai so |Q - Q*|; hang 2-3: % trang thai chon dung (tren lo trinh toi uu / moi trang thai)."""
    fig, axes = plt.subplots(3, len(cfgs), figsize=(16, 10.6), sharex="col")
    for j, cfg in enumerate(cfgs):
        name = cfg["name"]
        for key in [k for k in P.ALGO_COLOR if f"{name}|{k}|0|ep" in data]:
            ep, err = stack(data, name, key, "q_err")
            _, route = stack(data, name, key, "pol_route")
            _, acc = stack(data, name, key, "pol_acc")
            for i, y in enumerate((err, 100 * route, 100 * acc)):
                m, sd = y.mean(0), y.std(0)
                lo, hi = m - sd, m + sd
                if i == 0:  # truc log khong ve duoc 0: sai so = 0 (Q dung bang Q*) ve o day truc
                    m, lo, hi = (np.maximum(v, ERR_FLOOR) for v in (m, lo, hi))
                axes[i, j].plot(ep, m, color=P.ALGO_COLOR[key], lw=2, label=P.ALGO_LABEL[key])
                axes[i, j].fill_between(ep, lo, hi, color=P.ALGO_COLOR[key], alpha=0.12, lw=0)
        axes[0, j].set_title(P.TITLE[P.short(name)], loc="left", fontsize=10, color=P.INK)
        axes[0, j].set_yscale("log")
        axes[0, j].set_ylim(bottom=ERR_FLOOR)
        for i in (1, 2):
            axes[i, j].set_ylim(0, 102)
        axes[2, j].set_xlabel("Episode")
        kilo(axes[2, j])
    axes[0, 0].set_ylabel(P.L("Sai số trung bình |Q − Q*|\n(thang log; 0 vẽ ở đáy trục)",
                              "Mean error |Q − Q*|\n(log scale; 0 drawn at the bottom)"))
    axes[1, 0].set_ylabel(P.L("% chọn đúng trên các trạng thái\nthuộc lộ trình tối ưu",
                              "% optimal choices on states\nalong optimal routes"))
    axes[2, 0].set_ylabel(P.L("% chọn đúng trên mọi\ntrạng thái đạt tới", "% optimal choices on all\nreachable states"))
    handles, labels = [], []
    for ax in axes[0]:
        for h, lab in zip(*ax.get_legend_handles_labels()):
            if lab not in labels:
                handles.append(h)
                labels.append(lab)
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), bbox_to_anchor=(0.5, -0.01), fontsize=9.5)
    fig.suptitle(P.L("Hội tụ của bảng Q về Q* (Value Iteration), trung bình ± độ lệch chuẩn 5 seed",
                     "Convergence of the Q table to Q* (Value Iteration), mean ± std over 5 seeds"),
                 fontsize=13, fontweight="bold", color=P.INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    P.save(fig, "convergence.png")


def plot_start_q(cfg, data, seed=0):
    """Q(S, .) tai o xuat phat cua S2 theo thoi gian: Q-learning, SARSA, Monte Carlo ES (cung 1 seed)."""
    name = cfg["name"]
    qstar = data[f"{name}|qstar_start"]
    names = P.L(["Lên", "Xuống", "Trái", "Phải", "Pickup", "Drop"], ["Up", "Down", "Left", "Right", "Pickup", "Drop"])
    # 2 nuoc di hop le tu S (goc tren trai) va Pickup sai cho; mau tach biet voi mau thuat toan, da kiem tra CVD
    acts = [(DOWN, "#6250d6"), (RIGHT, "#e87ba4"), (PICKUP, "#c47f00")]
    keys = [k for k in P.ALGO_COLOR if f"{name}|{k}|{seed}|ep" in data]
    fig, axes = plt.subplots(1, len(keys), figsize=(5.6 * len(keys), 4.4), sharey=True)
    for ax, key in zip(np.atleast_1d(axes), keys):
        ep, q = data[f"{name}|{key}|{seed}|ep"], data[f"{name}|{key}|{seed}|q_start"]
        for a, col in acts:
            ax.plot(ep, q[:, a], color=col, lw=2)
            ax.axhline(qstar[a], color=col, lw=1.2, ls="--")
        a_end = int(np.argmax(q[-1]))
        ok = qstar[a_end] >= qstar.max() - 1e-6
        verdict = P.L("tối ưu" if ok else "sai", "optimal" if ok else "wrong")
        ax.set_title(f"{P.ALGO_LABEL[key]} · seed {seed}", loc="left", fontsize=10, color=P.INK)
        ax.text(0.98, 0.05, P.L(f"Cuối cùng robot chọn: {names[a_end]} ({verdict})",
                                f"Final greedy choice: {names[a_end]} ({verdict})"),
                transform=ax.transAxes, ha="right", va="bottom", fontsize=9.5, color=P.INK,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#c9c8c2"))
        ax.set_xlabel("Episode")
        kilo(ax)
    np.atleast_1d(axes)[0].set_ylabel("Q(S, a)")
    handles = [Line2D([], [], color=col, lw=2, label=names[a]) for a, col in acts]
    handles.append(Line2D([], [], color=P.INK2, lw=1.2, ls="--",
                          label=P.L("giá trị thật Q* (Value Iteration)", "true value Q* (Value Iteration)")))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), bbox_to_anchor=(0.5, -0.02), fontsize=9.5)
    fig.suptitle(P.L("S2 · Giá trị Q tại ô xuất phát S theo episode, so với giá trị thật Q* (nét đứt)",
                     "S2 · Q values at the start cell S over training vs the true Q* (dashed)"),
                 fontsize=12, fontweight="bold", color=P.INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    P.save(fig, "convergence_S2_start.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", choices=["vi", "en"], default="vi", help="ngon ngu nhan tren hinh")
    ap.add_argument("--plot-only", action="store_true", help="khong huan luyen lai, ve tu results/convergence.npz")
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
    d = np.load(OUT)
    data = {k: d[k] for k in d.files}
    print("Cuoi huan luyen (trung binh 5 seed): sai so |Q - Q*| | % chon dung tren lo trinh toi uu | tren moi trang thai")
    for c in cfgs:
        for key in [k for k in P.ALGO_COLOR if f"{c['name']}|{k}|0|ep" in data]:
            last = {m: stack(data, c["name"], key, m)[1][:, -1].mean() for m in ("q_err", "pol_route", "pol_acc")}
            print(f"  {P.short(c['name'])} {key:8s} {last['q_err']:9.2f} | {100 * last['pol_route']:5.1f}%"
                  f" | {100 * last['pol_acc']:5.1f}%")
    plot_convergence(cfgs, data)
    plot_start_q(next(c for c in cfgs if c["name"].startswith("S2")), data)


if __name__ == "__main__":
    main()
