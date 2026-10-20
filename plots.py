"""
plots.py - Ve tat ca bieu do tu results/ vao figures/.

    python plots.py              (chay sau experiments.py)
    python plots.py --lang en    (nhan tieng Anh, luu vao figures_en/)
"""
import argparse
import glob
import json
import os

import matplotlib
import matplotlib.ticker
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Circle, Rectangle  # noqa: E402

from agents import load_q, run_greedy  # noqa: E402
from delivery_env import (CARRYING, DELIVERED, DROP, MOVES, PICKUP, WAITING,  # noqa: E402
                          MultiDeliveryEnv, load_config)

FIG, RES = "figures", "results"
LANG = "vi"


def L(vi, en):
    """Chon chuoi theo ngon ngu cua hinh (doi bang --lang)."""
    return en if LANG == "en" else vi


# ----- Bang mau (mau chu luon la mau muc, mau du lieu chi o net/diem) -----
INK, INK2, GRIDC, WALL, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#3d3c39", "#ffffff"
# 3 slot dau cua bang mau phan loai (da kiem tra mu mau ca 3 cap); cau hinh cu ve mau xam trung tinh
ALGO_COLOR = {"ql": "#2a78d6", "sarsa": "#1baf7a", "mc": "#eb6834"}
ALGO_LABEL = {"ql": "Q-learning", "sarsa": "SARSA", "mc": "Monte Carlo ES"}
ALGO_TICK = {"ql": "Q-learning", "sarsa": "SARSA", "mc": "MC ES"}
OLD_COLOR = "#b8b6ae"
PARCEL_COLOR = {"A": "#6250d6", "B": "#e87ba4", "C": "#008300", "D": "#1baf7a"}  # luon kem chu cai cua kien
BOOST_COLOR = "#eda100"
TITLE = {"S1": "S1 · Cấp 1 (dễ): 2 kiện, mang được 2", "S2": "S2 · Cấp 2 (trung bình): 3 kiện, mang được 1",
         "S3": "S3 · Cấp 3 (khó): kiện C có deadline 20 bước",
         "S4": "S4 · Cấp 4 (rất khó): 4 kiện, ô ×2, D deadline 22"}
VARIANT_TITLE = {"S3_phat_tre_30": "Deadline 20, phạt trễ 30 (gốc)", "S3_phat_tre_0": "Deadline 20, phạt trễ 0",
                 "S3_deadline_45": "Deadline 45, phạt trễ 30", "S4_co_tang_tai": "Có ô tăng tải (gốc)",
                 "S4_khong_tang_tai": "Bỏ ô tăng tải"}
# ban tieng Anh cua cac nhan tren (main() ghi de khi --lang en)
ALGO_LABEL_EN = {}
TITLE_EN = {"S1": "S1 · Level 1 (easy): 2 parcels, carry 2", "S2": "S2 · Level 2 (medium): 3 parcels, carry 1",
            "S3": "S3 · Level 3 (hard): parcel C has a 20-step deadline",
            "S4": "S4 · Level 4 (very hard): 4 parcels, ×2 cell, D deadline 22"}
VARIANT_TITLE_EN = {"S3_phat_tre_30": "Deadline 20, late penalty 30 (base)",
                    "S3_phat_tre_0": "Deadline 20, late penalty 0",
                    "S3_deadline_45": "Deadline 45, late penalty 30", "S4_co_tang_tai": "With booster cell (base)",
                    "S4_khong_tang_tai": "Booster cell removed"}
EMPTY_PATH = "#8a8984"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": "#c9c8c2",
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRIDC, "grid.linewidth": 0.8, "axes.titleweight": "bold",
    "axes.titlesize": 11, "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
    "legend.frameon": False,
})


def configs_by_name():
    out = {}
    for p in sorted(glob.glob("configs/*.json")):
        c = load_config(p)
        out[c["name"]] = c
    return out


def short(name):
    return name.split("_")[0]


def kilo(ax):
    """Truc x ghi 20k, 1.5M thay vi 20000, 1500000."""
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
        lambda v, _: f"{v / 1e6:g}M" if abs(v) >= 1e6 else (f"{v / 1000:g}k" if abs(v) >= 1000 else f"{v:g}")))


def save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(os.path.join(FIG, name), facecolor=SURF)
    plt.close(fig)
    print("  ->", os.path.join(FIG, name))


# ======================================================================
# Ve luoi
# ======================================================================
def draw_grid(ax, env, status=None, boost=0):
    """Ve nen map: tuong, o S, o tang tai, diem lay (tron) va diem giao (vuong)."""
    status = status or (WAITING,) * env.N
    ax.set_xlim(-0.5, env.W - 0.5)
    ax.set_ylim(env.H - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    for r in range(env.H):
        for c in range(env.W):
            if (r, c) in env.walls:
                fc, ec = WALL, WALL
            elif (r, c) == env.booster:
                fc, ec = matplotlib.colors.to_rgba(BOOST_COLOR, 0.35 if boost == 0 else 0.1), "#c9c8c2"
            else:
                fc, ec = "#f6f5f2", "#d9d8d3"
            ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, fc=fc, ec=ec, lw=1))
    dr, dc = env.depot
    ax.text(dc - 0.36, dr - 0.3, "S", fontsize=8, color=INK2, fontweight="bold")
    if env.booster:
        br, bc = env.booster
        ax.text(bc, br, "×2", ha="center", va="center", fontsize=10, fontweight="bold", color=INK)
    for i, p in enumerate(env.parcels):
        col = PARCEL_COLOR[p["name"]]
        pr, pc = p["pickup"]
        if status[i] == WAITING:
            ax.add_patch(Circle((pc, pr), 0.3, fc=col, ec=SURF, lw=2, alpha=0.9))
            ax.text(pc, pr, p["name"].lower(), ha="center", va="center", color="white",
                    fontsize=10, fontweight="bold")
        qr, qc = p["drop"]
        done = status[i] == DELIVERED
        ax.add_patch(Rectangle((qc - 0.32, qr - 0.32), 0.64, 0.64, fc=col if not done else "none",
                               ec=col, lw=2, alpha=0.9 if not done else 0.5))
        ax.text(qc, qr, p["name"] if not done else "✓", ha="center", va="center",
                color="white" if not done else col, fontsize=10, fontweight="bold")


def map_legend(fig, has_booster=True):
    h = [Line2D([], [], marker="o", ls="", color=INK2, ms=9, label=L("Điểm lấy kiện (chữ thường)", "Pickup point (lowercase)")),
         Line2D([], [], marker="s", ls="", color=INK2, ms=9, label=L("Điểm giao kiện (chữ hoa)", "Drop point (uppercase)")),
         Rectangle((0, 0), 1, 1, fc=WALL, label=L("Tường", "Wall"))]
    if has_booster:
        h.append(Rectangle((0, 0), 1, 1, fc=matplotlib.colors.to_rgba(BOOST_COLOR, 0.35),
                           label=L("Ô tăng tải (×2)", "Booster cell (×2)")))
    fig.legend(handles=h, loc="lower center", ncol=len(h), bbox_to_anchor=(0.5, -0.02), fontsize=9)


# ======================================================================
# 1) Ban do 4 setting
# ======================================================================
def plot_maps(cfgs):
    fig, axes = plt.subplots(1, len(cfgs), figsize=(4.0 * len(cfgs), 4.9))
    for ax, (name, cfg) in zip(axes, cfgs.items()):
        env = MultiDeliveryEnv(cfg, "depot")
        draw_grid(ax, env)
        extra = [L(f"{env.N} kiện, sức chứa {env.capacity}", f"{env.N} parcels, capacity {env.capacity}")]
        if env.dl_idx is not None:
            pname = env.parcels[env.dl_idx]["name"]
            extra.append(L(f"{pname} deadline {env.deadline}", f"{pname} deadline {env.deadline}"))
        lv = cfg.get("level", "?")
        ax.set_title(L(f"{short(name)} · cấp {lv} · lưới {env.H}x{env.W}", f"{short(name)} · level {lv} · {env.H}x{env.W} grid")
                     + "\n" + ", ".join(extra), fontsize=10, color=INK)
    map_legend(fig)
    fig.suptitle(L("Bốn map, độ khó tăng dần", "Four maps of increasing difficulty"),
                 fontsize=13, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))  # chua cho cho suptitle
    save(fig, "maps.png")


# ======================================================================
# 2) Kich thuoc state space
# ======================================================================
def plot_state_space(summary):
    names = list(summary["settings"])
    full = [summary["settings"][n]["n_states"] for n in names]
    reach = [summary["settings"][n]["n_reachable"] for n in names]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(7.5, 4))
    w = 0.34
    b1 = ax.bar(x - w / 2 - 0.01, full, w, color="#b8b6ae",
                label=L("Kích thước mã hóa |S| = H·W·3^N·T·B", "Encoded size |S| = H·W·3^N·T·B"))
    b2 = ax.bar(x + w / 2 + 0.01, reach, w, color="#2a78d6",
                label=L("Trạng thái thực sự đạt tới (BFS)", "Actually reachable states (BFS)"))
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() * 1.08, f"{int(b.get_height()):,}",
                    ha="center", va="bottom", fontsize=9, color=INK)
    ax.set_yscale("log")
    ax.set_ylim(50, max(full) * 4)
    ax.set_xticks(x, [short(n) for n in names])
    ax.set_ylabel(L("Số trạng thái (thang log)", "Number of states (log scale)"))
    ax.set_title(L("Số trạng thái tăng theo cấp độ (deadline và ô tăng tải nhân thêm chiều τ, b)",
                   "State count grows with the level (deadline and booster add the τ, b dimensions)"),
                 loc="left", color=INK)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(axis="x", visible=False)
    save(fig, "state_space.png")


# ======================================================================
# 3) Duong hoc / ti le thanh cong / do dai episode
# ======================================================================
def _curves(summary, metric, ylabel, title, fname, ylim=None, ref=None):
    names = list(summary["settings"])
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    for ax, name in zip(axes.flat, names):
        data = np.load(os.path.join(RES, f"{name}.npz"))
        st = summary["settings"][name]
        for key in [k for k in ALGO_COLOR if f"{k}__eval_ep" in data]:
            # train_len: trung binh so buoc cua cac episode giua 2 lan danh gia
            arr = data[f"{key}__{metric}"].astype(float) * (100 if metric == "eval_acc" else 1)
            xs = data[f"{key}__eval_ep"]
            m, s = arr.mean(0), arr.std(0)
            ax.plot(xs, m, color=ALGO_COLOR[key], lw=2, label=ALGO_LABEL[key])
            if metric != "eval_success":
                lo, hi = (np.clip(m - s, 0, 100), np.clip(m + s, 0, 100)) if metric == "eval_acc" else (m - s, m + s)
                ax.fill_between(xs, lo, hi, color=ALGO_COLOR[key], alpha=0.12, lw=0)
        if ref == "optimal":
            y = st["optimal"]["return"]
            ax.axhline(y, color=INK2, lw=1)
            ax.text(xs[-1], y, L(f"tối ưu = {y:.0f}", f"optimal = {y:.0f}"), ha="right", va="bottom",
                    fontsize=9, color=INK2)
        elif ref == "max_steps":
            y = configs_by_name()[name]["max_steps"]
            ax.axhline(y, color=INK2, lw=1)
            ax.text(xs[-1], y, f"max_steps = {y}", ha="right", va="bottom", fontsize=9, color=INK2)
        ax.set_title(TITLE[short(name)], loc="left", fontsize=10, color=INK)
        kilo(ax)
        ax.set_xlabel("Episode")
        ax.set_ylabel(ylabel)
        if ylim:
            ax.set_ylim(*ylim)
        elif metric == "eval_return":  # cat phan rat am luc dau de thay ro vung gan toi uu
            ax.set_ylim(-1.15 * configs_by_name()[name]["max_steps"], st["optimal"]["return"] + 20)
    handles, labels = [], []
    for ax in axes.flat:
        for h, lab in zip(*ax.get_legend_handles_labels()):
            if lab not in labels:
                handles.append(h)
                labels.append(lab)
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), bbox_to_anchor=(0.5, -0.01), fontsize=9.5)
    fig.suptitle(title, fontsize=13, fontweight="bold", color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, fname)


# ======================================================================
# 4) Truc quan hoa policy
# ======================================================================
def _segments(traj):
    """Chia quy dao thanh cac doan theo 'pha' (trang thai kien + boost)."""
    phases = []
    for st in traj:
        key = (st[2], st[4])
        if not phases or phases[-1][0] != key:
            phases.append([key, []])
        phases[-1][1].append(st)
    return phases


def _carry_color(env, status):
    held = [env.parcels[i]["name"] for i, s in enumerate(status) if s == CARRYING]
    return PARCEL_COLOR[held[0]] if len(held) == 1 else (INK if held else EMPTY_PATH), held


def plot_policy(cfg, Q, fname, title):
    env = MultiDeliveryEnv(cfg, "depot")
    res = run_greedy(env, Q)
    phases = _segments(res["trajectory"])
    # bo pha cuoi neu chi la trang thai ket thuc (giao het)
    phases = [p for p in phases if not all(s == DELIVERED for s in p[0][0])]
    n = len(phases)
    cols = min(4, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.3 * cols, 3.9 * rows + 0.6))
    axes = np.atleast_1d(axes).ravel()
    for k, (ax, ((status, boost), seg)) in enumerate(zip(axes, phases)):
        draw_grid(ax, env, status, boost)
        tau = seg[0][3]
        # mui ten policy tham lam cho moi o trong pha nay
        for r in range(env.H):
            for c in range(env.W):
                if (r, c) in env.walls:
                    continue
                a = int(np.argmax(Q[env.encode((r, c, status, tau, boost))]))
                if a in MOVES:
                    dr, dc = MOVES[a]
                    ax.annotate("", xy=(c + 0.28 * dc, r + 0.28 * dr), xytext=(c - 0.22 * dc, r - 0.22 * dr),
                                arrowprops=dict(arrowstyle="-|>", color="#a3a29b", lw=1.1, mutation_scale=8))
                else:
                    ax.text(c + 0.3, r + 0.36, "P" if a == PICKUP else "D", fontsize=7,
                            color=INK2, ha="center", va="center", fontweight="bold")
        # duong di that su cua robot trong pha nay (noi them diem dau cua pha sau)
        pts = [(s[1], s[0]) for s in seg]
        if k + 1 < len(phases):
            nxt = phases[k + 1][1][0]
            pts.append((nxt[1], nxt[0]))
        color, held = _carry_color(env, status)
        xs, ys = zip(*pts)
        ax.plot(xs, ys, color=color, lw=2.5, solid_capstyle="round", zorder=5)
        ax.add_patch(Circle(pts[0], 0.16, fc=color, ec=SURF, lw=2, zorder=6))
        label = L("đang mang ", "carrying ") + "+".join(held) if held else L("tay không", "empty-handed")
        if env.booster:
            label += [L("", ""), L(" · tăng tải BẬT", " · booster ON"), L(" · tăng tải hết", " · booster used")][boost]
        ax.set_title(L(f"Pha {k + 1}: {label}", f"Phase {k + 1}: {label}"), fontsize=9.5, color=INK)
    for ax in axes[n:]:
        ax.axis("off")
    order = [e.split()[2] for _, e in res["events"] if e.startswith("giao kien")]
    fig.suptitle(L(f"{title}: giao {' → '.join(order)} trong {res['steps']} bước, return {res['return']:.0f}",
                   f"{title}: delivers {' → '.join(order)} in {res['steps']} steps, return {res['return']:.0f}"),
                 fontsize=12, fontweight="bold", color=INK, x=0.01, ha="left")
    map_legend(fig, env.booster is not None)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95), h_pad=2.0)
    save(fig, fname)


def draw_route(ax, env, Q, title):
    """Ve toan bo duong di 1 episode tham lam, to mau theo kien dang mang."""
    res = run_greedy(env, Q)
    draw_grid(ax, env)
    traj = res["trajectory"]
    phase = np.cumsum([0] + [a[2] != b[2] for a, b in zip(traj[:-1], traj[1:])])
    n_ph = phase[-1] + 1
    off = lambda k: (k - (n_ph - 1) / 2) * 0.07  # lech nhe moi pha de cac doan di lai khong chong nhau
    for t, (a, b) in enumerate(zip(traj[:-1], traj[1:])):
        color, _ = _carry_color(env, a[2])
        d = off(phase[t])
        ax.plot([a[1] + d, b[1] + d], [a[0] + d, b[0] + d], color=color, lw=2.5,
                solid_capstyle="round", zorder=5)
    ax.add_patch(Circle((traj[0][1], traj[0][0]), 0.16, fc=EMPTY_PATH, ec=SURF, lw=2, zorder=6))
    for k, (step, ev) in enumerate([e for e in res["events"] if e[1].startswith(("lay", "giao"))], 1):
        r, c = traj[step][0], traj[step][1]
        ax.add_patch(Circle((c + 0.3, r - 0.3), 0.17, fc="white", ec=INK, lw=1, zorder=7))
        ax.text(c + 0.3, r - 0.3, str(k), ha="center", va="center", fontsize=7.5, color=INK, zorder=8)
    order = [e.split()[2] for _, e in res["events"] if e.startswith("giao kien")]
    late = any("tre deadline" in e for _, e in res["events"])
    sub = L(f"thứ tự {'→'.join(order)} · {res['steps']} bước · return {res['return']:.0f}",
            f"order {'→'.join(order)} · {res['steps']} steps · return {res['return']:.0f}")
    if env.dl_idx is not None:
        sub += L(" · TRỄ hạn", " · LATE") if late else L(" · đúng hạn", " · on time")
    ax.set_title(f"{title}\n{sub}", fontsize=9.5, color=INK)
    return res


def plot_variants(summary):
    var = summary["variants"]
    saved = var["S4_khong_tang_tai"]["optimal"]["steps"] - var["S4_co_tang_tai"]["optimal"]["steps"]
    groups = [("S3", "reward_deadline.png",
               L("S3: phạt trễ và độ dài deadline quyết định giao kiện C trước hay sau",
                 "S3: the late penalty and deadline length decide whether C is delivered first or last")),
              ("S4", "env_booster.png",
               L(f"S4: có ô tăng tải thì robot mang 2 kiện một chuyến, ngắn hơn {saved} bước",
                 f"S4: with the booster the robot carries 2 parcels in one trip, {saved} steps shorter"))]
    for prefix, fname, title in groups:
        names = [v for v in var if v.startswith(prefix)]
        fig, axes = plt.subplots(1, len(names), figsize=(4.2 * len(names), 4.8))
        for ax, v in zip(np.atleast_1d(axes), names):
            cfg = load_config(os.path.join(RES, f"config_{v}.json"))
            env = MultiDeliveryEnv(cfg, "depot")
            Q = load_q(os.path.join(RES, f"Q_variant_{v}.npz"))
            draw_route(ax, env, Q, VARIANT_TITLE.get(v, var[v]["desc"]))
        handles = [Line2D([], [], color=PARCEL_COLOR[p["name"]], lw=2.5,
                          label=L(f"đang mang {p['name']}", f"carrying {p['name']}")) for p in env.parcels]
        handles.append(Line2D([], [], color=EMPTY_PATH, lw=2.5, label=L("tay không", "empty-handed")))
        if env.booster or env.capacity > 1:
            handles.append(Line2D([], [], color=INK, lw=2.5, label=L("mang 2 kiện", "carrying 2 parcels")))
        fig.legend(handles=handles, loc="lower center", ncol=len(handles), bbox_to_anchor=(0.5, -0.03), fontsize=9)
        fig.suptitle(title + L(" (số tròn = thứ tự sự kiện lấy/giao)", " (circled numbers = order of pickups/drops)"),
                     fontsize=12, fontweight="bold", color=INK, x=0.01, ha="left")
        fig.tight_layout(rect=(0, 0.03, 1, 0.92))
        save(fig, fname)


def plot_final_bars(summary):
    """Return cuoi cung cua policy tham lam: trung binh 5 seed + tung seed."""
    names = list(summary["settings"])
    fig, axes = plt.subplots(1, len(names), figsize=(12, 3.8), sharey=False)
    for ax, name in zip(axes, names):
        st = summary["settings"][name]
        keys = [k for k in ALGO_COLOR if k in st["algos"]]
        all_rets = [[r["return"] for r in st["algos"][k]["final_routes"]] for k in keys]
        lo = min(0, min(min(r) for r in all_rets))
        hi = max(st["optimal"]["return"], max(max(r) for r in all_rets))
        pad = 0.18 * (hi - lo)
        ax.set_ylim(lo - 0.05 * (hi - lo), hi + pad)  # chua cho cho nhan so phia tren cot
        for j, (key, rets) in enumerate(zip(keys, all_rets)):
            ax.bar(j, np.mean(rets), 0.6, color=ALGO_COLOR[key])
            ax.scatter(np.full(len(rets), j) + np.linspace(-0.15, 0.15, len(rets)), rets,
                       s=14, color=INK, zorder=5)
            ax.text(j, max(np.mean(rets), max(rets), 0) + 0.05 * (hi - lo), f"{np.mean(rets):.0f}",
                    ha="center", fontsize=9, color=INK)
        ax.axhline(st["optimal"]["return"], color=INK2, lw=1)
        ax.axhline(0, color="#c9c8c2", lw=0.8)
        ax.set_xticks(range(len(keys)), [ALGO_TICK[k] for k in keys])
        opt = st["optimal"]["return"]
        ax.set_title(L(f"{short(name)} (tối ưu {opt:.0f})", f"{short(name)} (optimal {opt:.0f})"),
                     fontsize=10, color=INK)
        ax.grid(axis="x", visible=False)
    n = len(summary["seeds"])
    axes[0].set_ylabel(L(f"Return policy cuối ({n} seed)", f"Final policy return ({n} seeds)"))
    fig.suptitle(L("Return của policy cuối: cột = trung bình, chấm = từng seed, vạch ngang = tối ưu",
                   "Final policy return: bar = mean, dots = individual seeds, line = optimal"),
                 fontsize=12, fontweight="bold", color=INK, x=0.01, ha="left")
    fig.tight_layout()
    save(fig, "final_returns.png")


def plot_ablation(summary):
    """Do chinh xac cuoi: cau hinh cu (xam) va cau hinh moi (mau thuat toan), cung so episode."""
    names = list(summary["settings"])
    if not summary["settings"][names[0]].get("baseline"):
        return
    fig, axes = plt.subplots(1, len(ALGO_COLOR), figsize=(13, 4.1), sharey=True)
    x, w = np.arange(len(names)), 0.36
    for ax, key in zip(axes, ALGO_COLOR):
        for dx, group, k, color, lab in [(-w / 2 - 0.01, "baseline", f"{key}_cu", OLD_COLOR, L("cũ", "old")),
                                         (w / 2 + 0.01, "algos", key, ALGO_COLOR[key], L("mới", "new"))]:
            accs = [100 * np.array(summary["settings"][n][group][k]["accuracy"]) for n in names]
            ax.bar(x + dx, [a.mean() for a in accs], w, color=color)
            for i, a in enumerate(accs):
                ax.scatter(np.full(len(a), x[i] + dx) + np.linspace(-0.08, 0.08, len(a)), a,
                           s=10, color=INK, zorder=5)
                ax.text(x[i] + dx, max(a.mean(), a.max()) + 3, f"{a.mean():.0f}", ha="center", fontsize=8.5,
                        color=INK)  # dat tren ca cot lan cac cham de khong de len nhau
        ax.set_xticks(x, [short(n) for n in names])
        ax.set_ylim(0, 114)
        ax.set_title(ALGO_LABEL[key], loc="left", fontsize=10.5, color=INK)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel(L("% ô xuất phát đi đúng lộ trình tối ưu", "% start cells on an optimal route"))
    handles = [Rectangle((0, 0), 1, 1, fc=OLD_COLOR, label=L("Cấu hình cũ", "Old settings"))]
    handles += [Rectangle((0, 0), 1, 1, fc=ALGO_COLOR[k], label=L(f"{ALGO_LABEL[k]}, cấu hình mới", f"{ALGO_LABEL[k]}, new settings"))
                for k in ALGO_COLOR]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), bbox_to_anchor=(0.5, -0.03), fontsize=9.5)
    fig.suptitle(L("Độ chính xác trước và sau cải tiến (cùng số episode; cột = trung bình, chấm = từng seed)",
                   "Accuracy before and after the improvements (same episodes; bar = mean, dots = seeds)"),
                 fontsize=12, fontweight="bold", color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, "ablation.png")


def main():
    global FIG, LANG
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", choices=["vi", "en"], default="vi", help="ngon ngu nhan tren hinh")
    LANG = ap.parse_args().lang
    if LANG == "en":
        FIG = "figures_en"
        ALGO_LABEL.update(ALGO_LABEL_EN)
        TITLE.update(TITLE_EN)
        VARIANT_TITLE.update(VARIANT_TITLE_EN)
    with open(os.path.join(RES, "summary.json"), encoding="utf-8") as f:
        summary = json.load(f)
    cfgs = configs_by_name()
    print("Ve bieu do:")
    plot_maps(cfgs)
    plot_state_space(summary)
    _curves(summary, "eval_return", L("Return của policy tham lam", "Greedy policy return"),
            L("Đường học: return của policy tham lam từ ô S (trung bình ± độ lệch chuẩn 5 seed; trục y cắt ở −1,15·max_steps)",
              "Learning curves: greedy policy return from S (mean ± std over 5 seeds; y-axis cut at −1.15·max_steps)"),
            "learning_curves.png", ref="optimal")
    _curves(summary, "eval_acc", L("% ô xuất phát đi đúng lộ trình tối ưu", "% start cells on an optimal route"),
            L("Độ chính xác: % ô xuất phát mà policy tham lam đạt return tối ưu (trung bình ± độ lệch chuẩn các seed)",
              "Accuracy: % start cells where the greedy policy reaches the optimal return (mean ± std over seeds)"),
            "accuracy.png", ylim=(-3, 103))
    _curves(summary, "eval_success", L("Tỉ lệ seed giao đủ hàng", "Fraction of seeds delivering all"),
            L("Tỉ lệ seed mà policy tham lam giao đủ mọi kiện",
              "Fraction of seeds whose greedy policy delivers every parcel"), "success_rate.png", ylim=(-0.05, 1.05))
    _curves(summary, "train_len", L("Số bước / episode", "Steps / episode"),
            L("Độ dài episode khi huấn luyện (trung bình giữa 2 lần đánh giá)",
              "Training episode length (mean between two evaluations)"),
            "episode_length.png", ref="max_steps")
    plot_final_bars(summary)
    plot_ablation(summary)
    for name, tag in [("S2_3_kien_suc_chua_1", "S2"), ("S4_o_tang_tai", "S4"), ("S3_kien_deadline", "S3")]:
        best = cfgs[name]["train"]["best"]
        Q = load_q(os.path.join(RES, f"Q_{name}_{best}.npz"))
        plot_policy(cfgs[name], Q, f"policy_{tag}.png", f"{tag} · {ALGO_LABEL[best]}")
    plot_variants(summary)


if __name__ == "__main__":
    main()
