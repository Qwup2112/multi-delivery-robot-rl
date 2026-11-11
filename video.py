"""
video.py - Quay video demo: robot chay ca 4 man (S1 -> S4) bang policy da hoc, luu thanh MP4.

    python video.py                 # chu tieng Viet -> figures/demo_all_levels.mp4
    python video.py --lang en       # chu tieng Anh  -> figures_en/demo_all_levels.mp4
    python video.py --algo sarsa    # policy cua SARSA (ql | sarsa | mc), mac dinh Q-learning
    python video.py --levels S3 S4  # chi quay mot so man

Moi man gom: 1 the gioi thieu (cap do, co che moi, lo trinh toi uu) roi robot chay tung buoc,
o ben phai la bang thong tin (buoc, return, trang thai tung kien, deadline, o tang tai, su kien).
Robot di chuyen muot (noi suy giua 2 o) va dung lai mot chut moi khi lay/giao kien.
Can ffmpeg (Colab va hau het Linux da co san). Bang Q doc tu results/ (chay experiments.py truoc);
neu chua co thi tu huan luyen nhanh nhu demo.py.
"""
import argparse
import os
import sys
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FFMpegWriter  # noqa: E402
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle  # noqa: E402

import plots  # noqa: E402
from agents import run_greedy  # noqa: E402
from delivery_env import CARRYING, DELIVERED, MultiDeliveryEnv  # noqa: E402
from demo import get_q  # noqa: E402
from planner import value_iteration  # noqa: E402

FPS = 12
SUB = 4           # so khung hinh cho moi buoc di chuyen (noi suy)
HOLD_EVENT = 8    # khung hinh dung lai khi lay/giao kien
HOLD_START = 10   # dung o trang thai dau
HOLD_END = 24     # dung o ket qua cuoi man
CARD = 30         # do dai the gioi thieu moi man

LEVEL = {
    "S1": ("Cấp 1 · Dễ", "Level 1 · Easy"),
    "S2": ("Cấp 2 · Trung bình", "Level 2 · Medium"),
    "S3": ("Cấp 3 · Khó", "Level 3 · Hard"),
    "S4": ("Cấp 4 · Rất khó", "Level 4 · Very hard"),
}
MECHANIC = {
    "S1": ("2 kiện, mang được cả 2 cùng lúc", "2 parcels, both can be carried together"),
    "S2": ("3 kiện, mỗi lần chỉ mang 1 → thứ tự giao quan trọng",
           "3 parcels, only 1 at a time → the order matters"),
    "S3": ("Kiện C phải giao trong 20 bước (trễ: −30)", "Parcel C must arrive within 20 steps (late: −30)"),
    "S4": ("4 kiện · kiện D deadline 22 bước · ô ×2 mang được 2 kiện cho 1 chuyến",
           "4 parcels · D has a 22-step deadline · the ×2 cell lets the robot carry 2 for one trip"),
}


def T(vi, en):
    return en if plots.LANG == "en" else vi


def event_text(ev):
    out = []
    for part in ev.split(" | "):
        part = part.strip()
        if part.startswith("lay kien"):
            out.append(T("lấy kiện ", "picked up ") + part.split()[-1])
        elif part.startswith("giao kien"):
            out.append(T("giao kiện ", "delivered ") + part.split()[-1])
        elif part == "kich hoat tang tai":
            out.append(T("bật ô ×2 (mang được 2)", "booster ON (carry 2)"))
        elif part == "tre deadline":
            out.append(T("trễ deadline (−30)", "deadline missed (−30)"))
        elif part == "pickup sai":
            out.append(T("lấy sai chỗ (−10)", "wrong Pickup (−10)"))
        elif part == "drop sai":
            out.append(T("giao sai chỗ (−10)", "wrong Drop (−10)"))
        elif part:
            out.append(part)
    return ", ".join(out)


def route_text(env, traj):
    """Thu tu giao hang, vd. D → C → B → A."""
    order, prev = [], traj[0][2]
    for st in traj[1:]:
        for i, (a, b) in enumerate(zip(prev, st[2])):
            if a != DELIVERED and b == DELIVERED:
                order.append(env.parcels[i]["name"])
        prev = st[2]
    return " → ".join(order)


class LevelClip:
    """Tinh truoc quy dao, phan thuong, su kien cua mot man roi sinh tung khung hinh."""

    def __init__(self, cfg, Q, algo):
        self.cfg, self.algo = cfg, algo
        self.key = cfg["name"].split("_")[0]
        self.env = MultiDeliveryEnv(cfg, "depot")
        res = run_greedy(self.env, Q)
        self.traj, self.ret, self.success = res["trajectory"], res["return"], res["success"]
        self.events = dict(res["events"])
        # phan thuong tung buoc (tinh lai bang ham chuyen tat dinh)
        self.rewards = []
        for a, b in zip(self.traj, self.traj[1:]):
            for act in range(6):
                s2, r, _, _ = self.env.transition(a, act)
                if s2 == b:
                    self.rewards.append(r)
                    break
        # return toi uu tu o S (Value Iteration) de so sanh
        Qopt, _ = value_iteration(MultiDeliveryEnv(cfg, "depot"))
        self.opt = run_greedy(MultiDeliveryEnv(cfg, "depot"), Qopt)["return"]
        self.route = route_text(self.env, self.traj)
        # danh sach khung hinh: (loai, buoc k, ti le noi suy u)
        self.frames = [("card", 0, 0.0)] * CARD + [("run", 0, 0.0)] * HOLD_START
        n = len(self.traj) - 1
        for k in range(n):
            moved = self.traj[k][:2] != self.traj[k + 1][:2]
            for j in range(1, SUB + 1 if moved else 2):
                self.frames.append(("run", k, j / (SUB if moved else 1)))
            if (k + 1) in self.events:
                self.frames += [("run", k + 1, 0.0)] * HOLD_EVENT
        self.frames += [("end", n, 0.0)] * HOLD_END

    # ----------------------------------------------------------------
    def draw_card(self, fig):
        fig.clf()
        fig.patch.set_facecolor("#16233f")
        lv = LEVEL[self.key][1 if plots.LANG == "en" else 0]
        mech = MECHANIC[self.key][1 if plots.LANG == "en" else 0]
        e = self.env
        fig.text(0.5, 0.70, f"{self.key} · {lv}", ha="center", va="center", fontsize=40,
                 fontweight="bold", color="white")
        fig.text(0.5, 0.56, mech, ha="center", va="center", fontsize=19, color="#f2b06b")
        info = T(f"Lưới {e.H}×{e.W} · {e.N} kiện · sức chứa {e.capacity} · tối đa {e.max_steps} bước",
                 f"{e.H}×{e.W} grid · {e.N} parcels · capacity {e.capacity} · at most {e.max_steps} steps")
        fig.text(0.5, 0.44, info, ha="center", va="center", fontsize=16, color="#d7dcea")
        alg = plots.ALGO_LABEL[self.algo]
        fig.text(0.5, 0.30, T(f"Policy đã học: {alg} (bảng Q, tham lam ε = 0)",
                              f"Learned policy: {alg} (Q table, greedy ε = 0)"),
                 ha="center", va="center", fontsize=15, color="#d7dcea")
        fig.text(0.5, 0.08, T("Topic 13 · Multi-Delivery Robot · Nguyễn Quang Trung · 22011211",
                              "Topic 13 · Multi-Delivery Robot · Nguyen Quang Trung · 22011211"),
                 ha="center", va="center", fontsize=11, color="#8f9bb8")

    # ----------------------------------------------------------------
    def draw_run(self, fig, k, u, final=False):
        fig.clf()
        fig.patch.set_facecolor("white")
        e, traj = self.env, self.traj
        st = traj[k]
        nxt = traj[min(k + 1, len(traj) - 1)]
        # ban do: trang thai kien da cap nhat khi den o tiep theo
        shown = nxt if u >= 1 else st
        ax = fig.add_axes([0.02, 0.06, 0.50, 0.86])
        plots.draw_grid(ax, e, shown[2], shown[4])
        # vet di
        upto = k + (1 if u >= 1 else 0)
        for a, b in zip(traj[:upto], traj[1:upto + 1]):
            col, _ = plots._carry_color(e, a[2])
            ax.plot([a[1], b[1]], [a[0], b[0]], color=col, lw=3, alpha=0.55, solid_capstyle="round", zorder=4)
        if 0 < u < 1:
            col, _ = plots._carry_color(e, st[2])
            r = st[0] + (nxt[0] - st[0]) * u
            c = st[1] + (nxt[1] - st[1]) * u
            ax.plot([st[1], c], [st[0], r], color=col, lw=3, alpha=0.55, solid_capstyle="round", zorder=4)
        else:
            r, c = shown[0], shown[1]
        col, held = plots._carry_color(e, st[2] if u < 1 else shown[2])
        ax.add_patch(Circle((c, r), 0.30, fc=col, ec="white", lw=2.5, zorder=8))
        ax.text(c, r, "+".join(held) if held else "", ha="center", va="center", color="white",
                fontsize=8 if len(held) > 1 else 10, fontweight="bold", zorder=9)

        # ---- bang thong tin ----
        step = k + (1 if u >= 1 else 0)
        cur = shown
        ret = sum(self.rewards[:step])
        x0 = 0.56
        lv = LEVEL[self.key][1 if plots.LANG == "en" else 0]
        fig.text(x0, 0.92, f"{self.key} · {lv}", fontsize=22, fontweight="bold", color=plots.INK)
        fig.text(x0, 0.875, textwrap.fill(MECHANIC[self.key][1 if plots.LANG == "en" else 0], 62), fontsize=11,
                 color=plots.INK2, va="top")
        fig.text(x0, 0.775, T("Bước", "Step"), fontsize=12, color=plots.INK2)
        fig.text(x0, 0.715, f"{step}", fontsize=28, fontweight="bold", color=plots.INK)
        fig.text(x0 + 0.13, 0.775, "Return", fontsize=12, color=plots.INK2)
        fig.text(x0 + 0.13, 0.715, f"{ret:+.0f}", fontsize=28, fontweight="bold",
                 color="#1a7f4b" if ret >= 0 else "#c0392b")
        fig.text(x0 + 0.26, 0.775, T("Thuật toán", "Algorithm"), fontsize=12, color=plots.INK2)
        fig.text(x0 + 0.26, 0.73, plots.ALGO_LABEL[self.algo], fontsize=15, fontweight="bold",
                 color=plots.ALGO_COLOR[self.algo])

        # trang thai tung kien
        y = 0.635
        fig.text(x0, y, T("Kiện hàng", "Parcels"), fontsize=12, fontweight="bold", color=plots.INK)
        names = {0: T("chờ lấy", "waiting"), 1: T("đang mang", "carried"), 2: T("đã giao ✓", "delivered ✓")}
        for i, p in enumerate(e.parcels):
            yy = y - 0.05 * (i + 1)
            fig.patches.append(Rectangle((x0, yy - 0.008), 0.018, 0.032, transform=fig.transFigure,
                                         fc=plots.PARCEL_COLOR[p["name"]], ec="none"))
            s = cur[2][i]
            dl = T(f" · deadline {p['deadline']}", f" · deadline {p['deadline']}") if p.get("deadline") else ""
            fig.text(x0 + 0.03, yy, f"{p['name']}: {names[s]}{dl}", fontsize=12,
                     color=plots.INK if s != DELIVERED else "#8a8984",
                     fontweight="bold" if s == CARRYING else "normal")
        y = y - 0.05 * (e.N + 1) - 0.01

        # deadline va o x2
        if e.dl_idx is not None:
            tau = cur[3]
            done_dl = cur[2][e.dl_idx] == DELIVERED
            frac = tau / e.deadline
            fig.text(x0, y, T(f"Đồng hồ deadline τ = {tau}/{e.deadline}", f"Deadline clock τ = {tau}/{e.deadline}")
                     + (T(" (đã giao)", " (delivered)") if done_dl else ""), fontsize=11, color=plots.INK)
            fig.patches.append(Rectangle((x0, y - 0.035), 0.36, 0.018, transform=fig.transFigure,
                                         fc="#ecebe7", ec="none"))
            late = tau >= e.deadline and not done_dl and step >= e.deadline
            fig.patches.append(Rectangle((x0, y - 0.035), 0.36 * frac, 0.018, transform=fig.transFigure,
                                         fc="#c0392b" if late else ("#1baf7a" if done_dl else "#eda100"), ec="none"))
            y -= 0.075
        if e.booster:
            b = cur[4]
            txt = [T("chưa dùng", "unused"), T("BẬT: mang được 2 kiện", "ON: can carry 2"),
                   T("đã dùng hết", "used up")][b]
            fig.text(x0, y, T("Ô ×2: ", "×2 cell: ") + txt, fontsize=11,
                     color=plots.INK, fontweight="bold" if b == 1 else "normal")
            y -= 0.05

        # nhat ky su kien
        fig.text(x0, y - 0.01, T("Sự kiện", "Events"), fontsize=12, fontweight="bold", color=plots.INK)
        evs = [(s, t) for s, t in sorted(self.events.items()) if s <= step][-5:]
        for j, (s, t) in enumerate(evs):
            fig.text(x0, y - 0.055 - 0.04 * j, T(f"bước {s}: ", f"step {s}: ") + event_text(t), fontsize=11,
                     color=plots.INK2 if j < len(evs) - 1 else plots.INK)

        if final:
            ok = abs(self.ret - self.opt) < 1e-9 and self.success
            msg = (T(f"Xong! {step} bước · return {self.ret:.0f} = tối ưu {self.opt:.0f}  ·  giao {self.route}",
                     f"Done! {step} steps · return {self.ret:.0f} = optimal {self.opt:.0f}  ·  order {self.route}")
                   if ok else T(f"Kết thúc: return {self.ret:.0f} (tối ưu {self.opt:.0f})",
                                f"Finished: return {self.ret:.0f} (optimal {self.opt:.0f})"))
            fig.patches.append(FancyBboxPatch((0.03, 0.005), 0.94, 0.045, transform=fig.transFigure,
                                              boxstyle="round,pad=0.004", fc="#e8f5ee" if ok else "#fdecea",
                                              ec="none"))
            fig.text(0.5, 0.027, msg, ha="center", va="center", fontsize=14, fontweight="bold",
                     color="#1a7f4b" if ok else "#c0392b")

    def render(self, fig, frame):
        kind, k, u = frame
        if kind == "card":
            self.draw_card(fig)
        else:
            self.draw_run(fig, k, u, final=(kind == "end"))


def outro(fig, clips):
    fig.clf()
    fig.patch.set_facecolor("#16233f")
    fig.text(0.5, 0.80, T("Kết quả cả 4 màn", "All four levels"), ha="center", fontsize=34,
             fontweight="bold", color="white")
    for j, c in enumerate(clips):
        lv = LEVEL[c.key][1 if plots.LANG == "en" else 0]
        ok = abs(c.ret - c.opt) < 1e-9 and c.success
        line = T(f"{c.key} · {lv}: {len(c.traj) - 1} bước, return {c.ret:.0f} (tối ưu {c.opt:.0f}), giao {c.route}",
                 f"{c.key} · {lv}: {len(c.traj) - 1} steps, return {c.ret:.0f} (optimal {c.opt:.0f}), order {c.route}")
        fig.text(0.5, 0.64 - 0.09 * j, ("✓ " if ok else "✗ ") + line, ha="center", fontsize=17,
                 color="#9fe0bd" if ok else "#f5a5a0")
    fig.text(0.5, 0.18, T("Q-learning, SARSA và Monte Carlo ES đều đạt 100% trên cả 4 màn (5/5 seed)",
                          "Q-learning, SARSA and Monte Carlo ES all reach 100% on the 4 levels (5/5 seeds)"),
             ha="center", fontsize=14, color="#d7dcea")
    fig.text(0.5, 0.08, "github.com/Qwup2112/multi-delivery-robot-rl", ha="center", fontsize=12, color="#8f9bb8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", choices=["vi", "en"], default="vi")
    ap.add_argument("--algo", choices=["ql", "sarsa", "mc"], default="ql")
    ap.add_argument("--levels", nargs="+", default=["S1", "S2", "S3", "S4"])
    ap.add_argument("--out", default=None, help="duong dan file MP4 (mac dinh figures[_en]/demo_all_levels.mp4)")
    args = ap.parse_args()
    plots.LANG = args.lang

    cfgs = {k.split("_")[0]: c for k, c in plots.configs_by_name().items()}
    clips = []
    for key in args.levels:
        cfg = cfgs[key]
        Q = get_q(cfg, args.algo)
        clip = LevelClip(cfg, Q, args.algo)
        print(f"{key}: {len(clip.traj) - 1} buoc, return {clip.ret:.0f} (toi uu {clip.opt:.0f}), "
              f"{len(clip.frames)} khung hinh")
        clips.append(clip)

    out = args.out or os.path.join("figures_en" if args.lang == "en" else "figures", "demo_all_levels.mp4")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
    writer = FFMpegWriter(fps=FPS, codec="libx264", bitrate=2500,
                          extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"])
    total = sum(len(c.frames) for c in clips) + 3 * FPS
    with writer.saving(fig, out, dpi=100):
        done = 0
        for clip in clips:
            for fr in clip.frames:
                clip.render(fig, fr)
                writer.grab_frame(facecolor=fig.get_facecolor())
                done += 1
                if done % 200 == 0:
                    print(f"  {done}/{total} khung hinh")
        outro(fig, clips)
        for _ in range(3 * FPS):
            writer.grab_frame(facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Da luu {out} ({total / FPS:.0f} giay)")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
