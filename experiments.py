"""
experiments.py - Chay toan bo thi nghiem, luu ket qua vao results/.

    python experiments.py               # day du: 5 seed moi thuat toan
    python experiments.py --quick       # thu nhanh: 2 seed, 1/20 so episode
    python experiments.py --scale 2     # gap doi so episode moi map (hoc lau hon -> chinh xac hon)
    python experiments.py --workers 2   # so process chay song song (mac dinh: so nhan CPU - 1)

Phan 1: moi map x 3 thuat toan (Q-learning, SARSA, Monte Carlo ES) x nhieu seed, cau hinh da cai tien.
Phan 2: doi reward / tham so moi truong (phat tre, do dai deadline, bo o tang tai).
Phan 3: cung so episode nhung dung cau hinh cu (truoc khi cai tien) -> do muc tang do chinh xac.
"""
import argparse
import copy
import glob
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

from agents import monte_carlo, q_learning, run_greedy, sarsa, save_q
from delivery_env import MultiDeliveryEnv, load_config
from planner import value_iteration

RESULTS = "results"
N_EVAL = 200  # so lan danh gia policy tham lam trong moi lan huan luyen

# Ten -> (nhan hien thi, ham hoc, tham so). Cau hinh da cai tien (do tham so tren 4 map, xem README muc 6):
#   exploring starts day du: episode bat dau o MOT TRANG THAI DAT TOI ngau nhien va hanh dong dau ngau nhien,
#   epsilon giam theo cap so nhan, moi thuat toan mot lich rieng.
_ES = {"start_mode": "exploring", "random_first": True}
ALGOS = {
    # moi truong tat dinh -> muc tieu r + gamma*max Q(s') khong nhieu, alpha = 1 hoc nhanh nhat
    "ql": ("Q-learning", q_learning, dict(_ES, alpha=1.0, eps_end=0.01, decay_frac=0.8, eps_decay="exp")),
    # muc tieu co nhieu (a' boc ngau nhien) -> alpha giam dan; eps ve gan 0 de hoc dung Q* (GLIE)
    "sarsa": ("SARSA", sarsa, dict(_ES, alpha=0.5, alpha_end=0.1, eps_end=1e-5, decay_frac=0.5, eps_decay="exp")),
    # Monte Carlo ES: policy tham lam (eps = 0), kham pha hoan toan nho exploring starts;
    # return cua policy tham lam la tat dinh nen lay return moi nhat (alpha = 1)
    "mc": ("Monte Carlo ES", monte_carlo, dict(_ES, alpha=1.0, eps_start=0.0, eps_end=0.0)),
}
# Cau hinh cu (truoc khi cai tien): xuat phat o trong ngau nhien, eps giam tuyen tinh 1 -> 0.05 trong 70% dau,
# Q0 = 0, alpha = 0.5 (TD), Monte Carlo trung binh mau 1/N.
_OLD = {"start_mode": "random_cell", "eps_end": 0.05, "decay_frac": 0.7}
BASELINE = {
    "ql_cu": ("Q-learning (cau hinh cu)", q_learning, dict(_OLD, alpha=0.5)),
    "sarsa_cu": ("SARSA (cau hinh cu)", sarsa, dict(_OLD, alpha=0.5)),
    "mc_cu": ("Monte Carlo (cau hinh cu)", monte_carlo, dict(_OLD)),
}
ALL = {**ALGOS, **BASELINE}


# Phan 2: (ten, file cau hinh goc, ham sua cau hinh, mo ta)
def _late0(c): c["rewards"]["late_penalty"] = 0
def _long_deadline(c): c["parcels"][2]["deadline"] = 45
def _no_booster(c): c["grid"] = [row.replace("B", ".") for row in c["grid"]]
def _same(c): pass


VARIANTS = [
    ("S3_phat_tre_30", "configs/s3_deadline.json", _same, "Deadline 20 buoc, phat tre 30 (goc)"),
    ("S3_phat_tre_0", "configs/s3_deadline.json", _late0, "Deadline 20 buoc, phat tre 0"),
    ("S3_deadline_45", "configs/s3_deadline.json", _long_deadline, "Deadline 45 buoc, phat tre 30"),
    ("S4_co_tang_tai", "configs/s4_booster.json", _same, "Co o tang tai (goc)"),
    ("S4_khong_tang_tai", "configs/s4_booster.json", _no_booster, "Bo o tang tai"),
]


def build_run(cfg, key, seed, scale=1.0):
    """Dung (ham hoc, env huan luyen, env danh gia, tham so) cho 1 lan chay. convergence.py dung lai."""
    tr = cfg["train"]
    _, fn, params = ALL[key]
    kw = dict(params)
    env = MultiDeliveryEnv(cfg, kw.pop("start_mode"))
    ev = MultiDeliveryEnv(cfg, "depot")
    n = max(N_EVAL, int(tr["episodes"] * scale))
    kw.update(n_episodes=n, gamma=tr["gamma"], seed=seed, eval_every=max(1, n // N_EVAL))
    return fn, env, ev, kw


def train_one(job):
    """Mot lan huan luyen (1 cau hinh, 1 thuat toan, 1 seed). Chay trong process con."""
    cfg, key, seed, scale = job
    fn, env, ev, kw = build_run(cfg, key, seed, scale)
    t0 = time.time()
    Q, hist = fn(env, ev, **kw)
    final = run_greedy(ev, Q)
    return {"name": cfg["name"], "algo": key, "seed": seed, "time": time.time() - t0,
            "hist": hist, "Q": Q if seed == 0 and key in ALGOS else None,
            "final": {k: final[k] for k in ("return", "success", "steps", "events")}}


def route_summary(res):
    """Thu tu giao kien + so buoc + co tre deadline khong."""
    order = [e.split()[2] for _, e in res["events"] if e.startswith("giao kien")]
    late = any("tre deadline" in e for _, e in res["events"])
    return {"return": res["return"], "steps": res["steps"], "success": res["success"],
            "order": "-".join(order), "late": late}


def optimal_info(cfg):
    ev = MultiDeliveryEnv(cfg, "depot")
    Qv, n_reach = value_iteration(ev, gamma=cfg["train"]["gamma"])
    return ev, Qv, n_reach, route_summary(run_greedy(ev, Qv))


def first_hit(eps, values, ok):
    """Episode danh gia dau tien thoa dieu kien ok(value) (None neu chua bao gio)."""
    return next((int(e) for e, v in zip(eps, values) if ok(v)), None)


def algo_stats(label, runs, opt_ret):
    finals = [o["final"] for o in runs]
    return {
        "label": label,
        "final_return_mean": float(np.mean([f["return"] for f in finals])),
        "final_return_std": float(np.std([f["return"] for f in finals])),
        "success_rate": float(np.mean([f["success"] for f in finals])),
        "optimal_rate": float(np.mean([f["return"] >= opt_ret - 1e-9 for f in finals])),
        # do chinh xac = % o xuat phat ma policy tham lam dat dung return toi uu (Value Iteration)
        "accuracy": [o["hist"]["eval_acc"][-1] for o in runs],
        "accuracy_mean": float(np.mean([o["hist"]["eval_acc"][-1] for o in runs])),
        "first_optimal_episode": [first_hit(o["hist"]["eval_ep"], o["hist"]["eval_return"],
                                            lambda r: r >= opt_ret - 1e-9) for o in runs],
        "first_full_accuracy_episode": [first_hit(o["hist"]["eval_ep"], o["hist"]["eval_acc"],
                                                  lambda a: a >= 1 - 1e-9) for o in runs],
        "final_steps": [f["steps"] for f in finals],
        "final_routes": [route_summary(f) for f in finals],
        "time_mean_s": float(np.mean([o["time"] for o in runs])),
    }


def warmup():
    """Bien dich numba mot lan o process chinh (luu cache) truoc khi chia viec cho cac process con."""
    cfg = load_config(sorted(glob.glob("configs/*.json"))[0])
    for key in ALGOS:
        fn, env, ev, kw = build_run(cfg, key, 0, scale=0)
        fn(env, ev, **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="2 seed, 1/20 so episode (chay thu)")
    ap.add_argument("--scale", type=float, default=1.0, help="nhan so episode cua moi map")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--no-baseline", action="store_true", help="bo qua Phan 3 (cau hinh cu)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args()
    seeds = list(range(2 if args.quick else args.seeds))
    scale = args.scale * (0.05 if args.quick else 1.0)
    os.makedirs(RESULTS, exist_ok=True)
    t_all = time.time()
    warmup()

    configs = [load_config(p) for p in sorted(glob.glob("configs/*.json"))]
    summary = {"seeds": seeds, "scale": scale, "settings": {}, "variants": {},
               "params": {k: {"label": v[0], **v[2]} for k, v in ALL.items()}}

    # ---------- Kich thuoc state space + policy toi uu (Value Iteration) ----------
    for cfg in configs:
        ev, Qv, n_reach, opt = optimal_info(cfg)
        summary["settings"][cfg["name"]] = {
            "title": cfg["title"], "level": cfg.get("level"), "grid": f"{ev.H}x{ev.W}", "n_parcels": ev.N,
            "capacity": ev.capacity, "n_pos": ev.n_pos, "n_codes": ev.n_codes,
            "n_tau": ev.n_tau, "n_boost": ev.n_boost, "n_states": ev.n_states,
            "n_reachable": n_reach, "n_actions": int(ev.action_space.n), "optimal": opt,
            "episodes": max(N_EVAL, int(cfg["train"]["episodes"] * scale)), "algos": {}, "baseline": {}}
        print(f"{cfg['name']}: |S| = {ev.n_states}, dat toi = {n_reach}, toi uu = {opt}", flush=True)

    # ---------- Phan 1 + Phan 3: huan luyen song song ----------
    keys = list(ALGOS) + ([] if args.no_baseline else list(BASELINE))
    jobs = [(cfg, key, s, scale) for cfg in configs for key in keys for s in seeds]
    jobs.sort(key=lambda j: -j[0]["train"]["episodes"] * (2 if j[1].startswith("mc") else 1))  # job dai chay truoc
    t0 = time.time()
    with Pool(args.workers) as pool:
        outs = []
        for i, out in enumerate(pool.imap_unordered(train_one, jobs), 1):
            outs.append(out)
            print(f"  [{i}/{len(jobs)}] {out['name']} {out['algo']} seed={out['seed']} "
                  f"return={out['final']['return']:.0f} acc={out['hist']['eval_acc'][-1]:.2f} "
                  f"({out['time']:.0f}s)", flush=True)
    print(f"Phan 1 + 3 xong sau {time.time() - t0:.0f}s", flush=True)

    for cfg in configs:
        name = cfg["name"]
        st = summary["settings"][name]
        arrays = {}
        for key in keys:
            runs = sorted([o for o in outs if o["name"] == name and o["algo"] == key], key=lambda o: o["seed"])
            for m in ("eval_return", "eval_success", "eval_acc", "train_return", "train_len"):
                arrays[f"{key}__{m}"] = np.array([o["hist"][m] for o in runs], dtype=float)
            arrays[f"{key}__eval_ep"] = np.array(runs[0]["hist"]["eval_ep"])
            group = "algos" if key in ALGOS else "baseline"
            st[group][key] = algo_stats(ALL[key][0], runs, st["optimal"]["return"])
            if key in ALGOS:
                save_q(os.path.join(RESULTS, f"Q_{name}_{key}.npz"), runs[0]["Q"])
        np.savez_compressed(os.path.join(RESULTS, f"{name}.npz"), **arrays)

    # ---------- Phan 2: doi reward / tham so moi truong ----------
    base = {os.path.normpath(p): load_config(p) for p in glob.glob("configs/*.json")}
    var_jobs, var_meta = [], {}
    for vname, path, modify, desc in VARIANTS:
        cfg = copy.deepcopy(base[os.path.normpath(path)])
        modify(cfg)
        cfg["name"] = vname
        _, _, n_reach, opt = optimal_info(cfg)
        var_meta[vname] = {"desc": desc, "base": path, "optimal": opt, "n_reachable": n_reach}
        var_jobs += [(cfg, cfg["train"]["best"], s, scale) for s in seeds]
        with open(os.path.join(RESULTS, f"config_{vname}.json"), "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=1)
    t0 = time.time()
    with Pool(args.workers) as pool:
        var_outs = pool.map(train_one, var_jobs)
    print(f"Phan 2 xong sau {time.time() - t0:.0f}s", flush=True)
    for vname, meta in var_meta.items():
        runs = sorted([o for o in var_outs if o["name"] == vname], key=lambda o: o["seed"])
        meta["algo"] = runs[0]["algo"]
        meta["learned_routes"] = [route_summary(o["final"]) for o in runs]
        meta["accuracy"] = [o["hist"]["eval_acc"][-1] for o in runs]
        save_q(os.path.join(RESULTS, f"Q_variant_{vname}.npz"), runs[0]["Q"])
        print(f"{vname}: toi uu {meta['optimal']} | hoc duoc {meta['learned_routes'][0]}", flush=True)
    summary["variants"] = var_meta

    # ---------- Bang tom tat ----------
    print("\nDo chinh xac cuoi (% o xuat phat di dung lo trinh toi uu, trung binh cac seed):")
    for name, st in summary["settings"].items():
        row = [f"{k} {100 * a['accuracy_mean']:5.1f}%" for g in ("algos", "baseline") for k, a in st[g].items()]
        print(f"  {name:22s} " + " | ".join(row))
    summary["time_total_s"] = time.time() - t_all
    with open(os.path.join(RESULTS, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print(f"Da luu results/summary.json (tong {summary['time_total_s']:.0f}s)")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
