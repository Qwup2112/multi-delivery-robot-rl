"""
demo.py - Chay MOT episode bang policy da hoc va in tung buoc ra man hinh.

    python demo.py                                         # mac dinh: S4, bien the Q-learning tot nhat
    python demo.py --config configs/s2_three_parcels.json  # chon map khac
    python demo.py --algo sarsa                            # xem policy cua SARSA (ql | sarsa | mc)
    python demo.py --gif                                   # luu them anh dong figures/demo_<map>.gif
    python demo.py --gif --lang en                         # anh dong chu tieng Anh, luu vao figures_en/

Neu chua co bang Q trong results/ (chua chay experiments.py), demo tu huan luyen thuat toan da chon
(cung tham so voi experiments.py, 1 seed).
"""
import argparse
import os
import sys
import time

import numpy as np

from agents import load_q, run_greedy
from delivery_env import ACTION_NAMES, MultiDeliveryEnv, load_config


def get_q(cfg, algo):
    path = os.path.join("results", f"Q_{cfg['name']}_{algo}.npz")
    if os.path.exists(path):
        print(f"Doc bang Q tu {path}")
        return load_q(path)
    from experiments import build_run
    fn, env, ev, kw = build_run(cfg, algo, seed=0)
    print(f"Chua co {path} -> huan luyen nhanh {algo}, {kw['n_episodes']} episode...")
    Q, _ = fn(env, ev, **kw)
    return Q


def play(cfg, Q, delay):
    env = MultiDeliveryEnv(cfg, "depot")
    s, _ = env.reset()
    print(f"\n=== {cfg['title']} ===")
    print(env.render())
    total, done = 0.0, False
    while not done:
        a = int(np.argmax(Q[s]))
        s, r, term, trunc, info = env.step(a)
        total += r
        done = term or trunc
        note = f"  <- {info['event']}" if info["event"] else ""
        print(f"\nBuoc {env.t}: {ACTION_NAMES[a]}, reward {r:+.0f}, tong {total:+.0f}{note}")
        print(env.render())
        time.sleep(delay)
    print("\nKET QUA:", "giao du hang" if term else "het max_steps (that bai)", f"| return = {total:.0f}")


def save_gif(cfg, Q, path, lang="vi"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.patches import Circle

    from plots import SURF, _carry_color, draw_grid

    env = MultiDeliveryEnv(cfg, "depot")
    res = run_greedy(env, Q)
    traj = res["trajectory"]
    fig, ax = plt.subplots(figsize=(4.2, 4.6))

    def frame(k):
        ax.clear()
        st = traj[k]
        draw_grid(ax, env, st[2], st[4])
        for a, b in zip(traj[:k], traj[1:k + 1]):
            color, _ = _carry_color(env, a[2])
            ax.plot([a[1], b[1]], [a[0], b[0]], color=color, lw=2, alpha=0.6)
        color, held = _carry_color(env, st[2])
        ax.add_patch(Circle((st[1], st[0]), 0.22, fc=color, ec=SURF, lw=2, zorder=6))
        if lang == "en":
            txt = f"Step {k}/{len(traj) - 1} · " + ("carrying " + "+".join(held) if held else "empty-handed")
        else:
            txt = f"Bước {k}/{len(traj) - 1} · " + ("mang " + "+".join(held) if held else "tay không")
        if env.dl_idx is not None:
            txt += f" · τ={st[3]}/{env.deadline}"
        ax.set_title(txt, fontsize=10)

    anim = FuncAnimation(fig, frame, frames=len(traj), interval=350)
    anim.save(path, writer=PillowWriter(fps=3))
    plt.close(fig)
    print("Da luu", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/s4_booster.json")
    ap.add_argument("--algo", default=None, help="ql | sarsa | mc (mac dinh: best trong config)")
    ap.add_argument("--delay", type=float, default=0.0, help="so giay nghi giua cac buoc")
    ap.add_argument("--gif", action="store_true")
    ap.add_argument("--lang", choices=["vi", "en"], default="vi", help="ngon ngu chu tren anh dong (en -> figures_en/)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    algo = args.algo or cfg["train"]["best"]
    Q = get_q(cfg, algo)
    play(cfg, Q, args.delay)
    if args.gif:
        out = "figures_en" if args.lang == "en" else "figures"
        os.makedirs(out, exist_ok=True)
        save_gif(cfg, Q, os.path.join(out, f"demo_{cfg['name'].split('_')[0]}.gif"), args.lang)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
