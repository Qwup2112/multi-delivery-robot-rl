"""
agents.py - Hai thuat toan dang bang (tabular): Q-learning va Monte Carlo first-visit.

Ca hai dung bang Q kich thuoc (so_trang_thai x 6) va chinh sach epsilon-greedy.
  Q-learning : Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]
  Monte Carlo: Q(s,a) <- Q(s,a) + (1/N(s,a)) * [G_t - Q(s,a)],  G_t = r_{t+1} + gamma*G_{t+1}
"""
import numpy as np


def epsilon_at(ep, n_episodes, eps_start, eps_end, decay_frac):
    """Epsilon giam tuyen tinh tu eps_start xuong eps_end trong decay_frac * n_episodes episode dau."""
    frac = min(1.0, ep / max(1, decay_frac * n_episodes))
    return eps_start + frac * (eps_end - eps_start)


def eps_greedy(Q, s, eps, rng):
    """Voi xac suat eps chon ngau nhien, nguoc lai chon a co Q lon nhat (hoa thi boc ngau nhien)."""
    if rng.random() < eps:
        return int(rng.integers(Q.shape[1]))
    q = Q[s]
    best = np.flatnonzero(q == q.max())
    return int(best[0]) if len(best) == 1 else int(rng.choice(best))


def run_greedy(env, Q, start=None):
    """Chay 1 episode voi policy tham lam (eps = 0). Tra ve dict ket qua + quy dao."""
    s, _ = env.reset(options={"start": start} if start else None)
    traj = [env.state]
    events, total, done, steps = [], 0.0, False, 0
    while not done:
        a = int(np.argmax(Q[s]))
        s, r, term, trunc, info = env.step(a)
        total += r
        steps += 1
        traj.append(env.state)
        if info["event"]:
            events.append((steps, info["event"]))
        done = term or trunc
    return {"return": total, "success": bool(term), "steps": steps,
            "trajectory": traj, "events": events}


def _new_history():
    return {"train_return": [], "train_len": [], "eval_ep": [], "eval_return": [], "eval_success": []}


def _log(hist, ep, G, length, eval_env, Q, eval_every, n_episodes):
    hist["train_return"].append(G)
    hist["train_len"].append(length)
    if ep % eval_every == 0 or ep == n_episodes - 1:
        res = run_greedy(eval_env, Q)
        hist["eval_ep"].append(ep)
        hist["eval_return"].append(res["return"])
        hist["eval_success"].append(res["success"])


def q_learning(env, eval_env, n_episodes, alpha=0.1, gamma=0.99,
               eps_start=1.0, eps_end=0.05, decay_frac=0.7, seed=0, eval_every=50, q_init=0.0):
    rng = np.random.default_rng(seed)
    env.reset(seed=seed)
    Q = np.full((env.n_states, env.action_space.n), float(q_init))
    hist = _new_history()
    for ep in range(n_episodes):
        eps = epsilon_at(ep, n_episodes, eps_start, eps_end, decay_frac)
        s, _ = env.reset()
        G, length, done = 0.0, 0, False
        while not done:
            a = eps_greedy(Q, s, eps, rng)
            s2, r, term, trunc, _ = env.step(a)
            # Neu ket thuc that (giao het) thi khong cong gia tri tuong lai.
            # Neu chi bi cat vi het max_steps (truncated) van bootstrap binh thuong.
            target = r if term else r + gamma * Q[s2].max()
            Q[s, a] += alpha * (target - Q[s, a])
            s = s2
            G += r
            length += 1
            done = term or trunc
        _log(hist, ep, G, length, eval_env, Q, eval_every, n_episodes)
    return Q, hist


def monte_carlo(env, eval_env, n_episodes, gamma=0.99, alpha=None, first_visit=True,
                eps_start=1.0, eps_end=0.05, decay_frac=0.7, seed=0, eval_every=50, q_init=0.0):
    """On-policy MC control.
    alpha=None -> trung binh mau (1/N); alpha=so -> buoc hoc co dinh.
    first_visit=True -> chi cap nhat lan dau (s,a) xuat hien; False -> every-visit."""
    rng = np.random.default_rng(seed)
    env.reset(seed=seed)
    Q = np.full((env.n_states, env.action_space.n), float(q_init))
    N = np.zeros_like(Q)
    hist = _new_history()
    for ep in range(n_episodes):
        eps = epsilon_at(ep, n_episodes, eps_start, eps_end, decay_frac)
        # 1) Sinh tron mot episode bang policy hien tai
        s, _ = env.reset()
        episode, done = [], False
        while not done:
            a = eps_greedy(Q, s, eps, rng)
            s2, r, term, trunc, _ = env.step(a)
            episode.append((s, a, r))
            s = s2
            done = term or trunc
        # 2) Lan dau tien moi cap (s,a) xuat hien
        first = {}
        for t, (s, a, _) in enumerate(episode):
            first.setdefault((s, a), t)
        # 3) Tinh return nguoc tu cuoi episode va cap nhat Q
        G = 0.0
        for t in range(len(episode) - 1, -1, -1):
            s, a, r = episode[t]
            G = gamma * G + r
            if not first_visit or first[(s, a)] == t:
                N[s, a] += 1
                step = alpha if alpha else 1.0 / N[s, a]
                Q[s, a] += step * (G - Q[s, a])
        _log(hist, ep, sum(x[2] for x in episode), len(episode), eval_env, Q, eval_every, n_episodes)
    return Q, hist
