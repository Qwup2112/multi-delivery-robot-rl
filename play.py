"""
play.py - Tu dieu khien robot bang ban phim (nguoi choi thay cho agent).

    python play.py                                    # mac dinh: S1 (de nhat)
    python play.py --config configs/s4_booster.json   # chon map khac

Phim:  W / A / S / D = len / trai / xuong / phai
       J = nhat kien (Pickup), K = giao kien (Drop)
       R = choi lai, Q = thoat
Luat choi va reward giong het luc agent hoc (delivery_env.py). Het van se so voi lo trinh toi uu
(Value Iteration). Neu cua so khong doc duoc tung phim (Colab, Git Bash), go chuoi phim roi Enter, vd: ddsj
"""
import argparse
import os
import sys

from delivery_env import ACTION_NAMES, DOWN, DROP, LEFT, MOVES, PICKUP, RIGHT, UP, MultiDeliveryEnv, load_config

KEYS = {"w": UP, "a": LEFT, "s": DOWN, "d": RIGHT, "j": PICKUP, "k": DROP}
LEGEND = "@ = robot, a/b/c/d = kien cho lay, A/B/C/D = dich, * = da giao, + = o tang tai, # = tuong"


def read_key():
    """Doc 1 phim, khong can Enter."""
    if os.name == "nt":
        import msvcrt
        ch = msvcrt.getwch()
        if ch in ("\x00", "\xe0"):  # phim dac biet (mui ten, F1, ...) -> bo qua
            msvcrt.getwch()
            return ""
        return ch.lower()
    import termios
    import tty
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        return sys.stdin.read(1).lower()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def key_stream():
    """Cac phim nguoi choi bam. Khong co terminal that (Colab, Git Bash, pipe) -> doc ca dong roi tach tung phim."""
    if sys.stdin.isatty():
        while True:
            yield read_key()
    while True:
        try:
            line = input("Phim (vd: ddsj) roi Enter: ")
        except EOFError:
            yield "q"
            return
        yield from line.strip().lower()


def optimal_text(cfg):
    from experiments import optimal_info  # Value Iteration, chi chay khi het van
    opt = optimal_info(cfg)[3]
    return f"Toi uu (Value Iteration): return {opt['return']:.0f}, {opt['steps']} buoc, thu tu giao {opt['order']}"


def play(cfg, keys, clear):
    """Choi 1 van tu o S. Tra ve True neu nguoi choi bam R (choi lai)."""
    env = MultiDeliveryEnv(cfg, "depot")
    env.reset()
    total, done, msg = 0.0, False, "Bat dau tu o S."
    while True:
        if clear:
            os.system("cls" if os.name == "nt" else "clear")
        print(f"=== {cfg['title']} ===\n{LEGEND}\n")
        print(env.render())
        print(f"\n{msg}")
        print("R: choi lai | Q: thoat" if done else "W/A/S/D: di chuyen | J: nhat | K: giao | R: choi lai | Q: thoat")

        key = next(keys)
        if key in ("q", "\x03"):  # \x03 = Ctrl+C khi doc tung phim
            return False
        if key == "r":
            return True
        if done or key not in KEYS:
            continue

        a = KEYS[key]
        before = env.state
        _, r, term, trunc, info = env.step(a)
        total += r
        notes = [info["event"]] if info["event"] else []
        if a in MOVES and env.state[:2] == before[:2]:
            notes.insert(0, "dam tuong / ra ngoai -> dung yen")
        msg = f"Buoc {env.t}: {ACTION_NAMES[a]}, reward {r:+.0f}, tong {total:+.0f}"
        if notes:
            msg += "  <- " + " | ".join(notes)
        if term or trunc:
            done = True
            msg += "\n\nKET QUA: " + ("giao du hang" if term else f"het {cfg['max_steps']} buoc (that bai)")
            msg += f" | return = {total:.0f}\n" + optimal_text(cfg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/s1_two_parcels.json")
    args = ap.parse_args()
    cfg = load_config(args.config)
    keys = key_stream()
    while play(cfg, keys, clear=sys.stdin.isatty()):
        pass


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
