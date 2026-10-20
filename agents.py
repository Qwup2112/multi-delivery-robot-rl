"""
agents.py - Ba thuat toan dang bang (tabular): Q-learning, SARSA va Monte Carlo first-visit.

Ca ba dung bang Q kich thuoc (so_trang_thai x 6) va chinh sach epsilon-greedy.
  Q-learning : Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]   (off-policy)
  SARSA      : Q(s,a) <- Q(s,a) + alpha * [r + gamma * Q(s',a')        - Q(s,a)]   (on-policy,
               a' la hanh dong agent THAT SU chon o s' bang epsilon-greedy)
  Monte Carlo: Q(s,a) <- Q(s,a) + step * [G_t - Q(s,a)],  G_t = r_{t+1} + gamma*G_{t+1}
               step = 1/N(s,a) (trung binh mau) hoac alpha co dinh

Tang toc: moi truong tat dinh nen env.tables() lap san ham chuyen thanh 3 mang NS, R, D.
Vong lap huan luyen chi tra mang va duoc bien dich bang numba (@njit) -> nhanh hon rat nhieu
so voi goi env.step() bang Python. Khong co numba thi code van chay (Python thuan, cham hon).
"""
import numpy as np

try:
    from numba import njit
except ImportError:  # khong co numba: dung ham Python binh thuong
    def njit(*args, **kwargs):
        return args[0] if args and callable(args[0]) else (lambda f: f)

QL, SARSA = 0, 1  # chon cong thuc muc tieu trong _td_chunk


# ======================================================================
# Cac ham nho dung chung (bien dich bang numba)
# ======================================================================
@njit(cache=True)
def _seed(seed):
    np.random.seed(seed)  # seed bo sinh so ngau nhien ben trong numba


@njit(cache=True)
def _schedule(ep, n_episodes, start, end, decay_frac, exponential):
    """Gia tri tai episode ep: di tu start toi end trong decay_frac * n_episodes episode dau roi giu nguyen.
    exponential=True -> giam theo cap so nhan (nhanh luc dau, cham dan ve sau), nguoc lai giam tuyen tinh."""
    frac = min(1.0, ep / max(1.0, decay_frac * n_episodes))
    if exponential:
        return start * (end / start) ** frac
    return start + frac * (end - start)


@njit(cache=True)
def _argmax(Q, s):
    """Hanh dong co Q lon nhat (hoa thi lay chi so dau, giong np.argmax)."""
    best = 0
    for a in range(1, Q.shape[1]):
        if Q[s, a] > Q[s, best]:
            best = a
    return best


@njit(cache=True)
def _eps_greedy(Q, s, eps):
    """Voi xac suat eps chon ngau nhien, nguoc lai chon a co Q lon nhat (hoa thi boc ngau nhien deu)."""
    if np.random.random() < eps:
        return np.random.randint(Q.shape[1])
    best, choice, ties = Q[s, 0], 0, 1
    for a in range(1, Q.shape[1]):
        if Q[s, a] > best:
            best, choice, ties = Q[s, a], a, 1
        elif Q[s, a] == best:
            ties += 1
            if np.random.randint(ties) == 0:  # giu moi ung vien voi xac suat 1/ties -> deu nhau
                choice = a
    return choice


@njit(cache=True)
def _rollouts(Q, NS, R, D, starts, max_steps, out_ret, out_len, out_done):
    """Chay policy tham lam (eps = 0) tu tung trang thai trong starts, ghi return / so buoc / co giao het."""
    for i in range(starts.shape[0]):
        s, G, done, t = starts[i], 0.0, False, 0
        while t < max_steps and not done:
            a = _argmax(Q, s)
            G += R[s, a]
            done = D[s, a]
            s = NS[s, a]
            t += 1
        out_ret[i], out_len[i], out_done[i] = G, t, done


# ======================================================================
# Vong lap huan luyen (moi lan goi chay cac episode ep0..ep1-1)
# ======================================================================
@njit(cache=True)
def _td_chunk(Q, NS, R, D, starts, max_steps, method, gamma, ep0, ep1, n_episodes,
              alpha0, alpha1, random_first, eps0, eps1, decay_frac, eps_exp, out_ret, out_len):
    """Q-learning (method = QL) hoac SARSA (method = SARSA): cap nhat sau MOI buoc."""
    for ep in range(ep0, ep1):
        eps = _schedule(ep, n_episodes, eps0, eps1, decay_frac, eps_exp)
        alpha = _schedule(ep, n_episodes, alpha0, alpha1, 1.0, False)  # alpha giam tuyen tinh (neu alpha1 < alpha0)
        s = starts[np.random.randint(starts.shape[0])]
        # random_first: hanh dong dau tien chon ngau nhien deu (exploring starts cho ca cap (s, a))
        a = np.random.randint(Q.shape[1]) if random_first else _eps_greedy(Q, s, eps)
        a2, G, t = 0, 0.0, 0
        while t < max_steps:
            s2, r, term = NS[s, a], R[s, a], D[s, a]
            G += r
            t += 1
            if term:  # giao het hang: khong con gia tri tuong lai
                target = r
            elif method == SARSA:
                a2 = _eps_greedy(Q, s2, eps)  # hanh dong se lam that o s' (co the la buoc kham pha)
                target = r + gamma * Q[s2, a2]
            else:
                target = r + gamma * Q[s2, _argmax(Q, s2)]  # gia tri hanh dong TOT NHAT o s'
            # Neu chi bi cat vi het max_steps (truncated) van bootstrap binh thuong.
            Q[s, a] += alpha * (target - Q[s, a])
            if term:
                break
            s = s2
            a = a2 if method == SARSA else _eps_greedy(Q, s, eps)
        out_ret[ep - ep0], out_len[ep - ep0] = G, t


@njit(cache=True)
def _mc_chunk(Q, N, stamp, NS, R, D, starts, max_steps, gamma, ep0, ep1, n_episodes, alpha,
              first_visit, boot_trunc, random_first, eps0, eps1, decay_frac, eps_exp, buf_s, buf_a, buf_r,
              buf_first, out_ret, out_len):
    """Monte Carlo on-policy: sinh tron episode roi moi cap nhat Q tu return G."""
    for ep in range(ep0, ep1):
        eps = _schedule(ep, n_episodes, eps0, eps1, decay_frac, eps_exp)
        # 1) Sinh tron mot episode bang policy hien tai
        s, T, term, total = starts[np.random.randint(starts.shape[0])], 0, False, 0.0
        while T < max_steps and not term:
            if T == 0 and random_first:  # Monte Carlo ES: hanh dong dau tien chon ngau nhien deu
                a = np.random.randint(Q.shape[1])
            else:
                a = _eps_greedy(Q, s, eps)
            buf_s[T], buf_a[T], buf_r[T] = s, a, R[s, a]
            total += R[s, a]
            term = D[s, a]
            s = NS[s, a]
            T += 1
        # 2) Danh dau lan dau tien moi cap (s,a) xuat hien (stamp ghi so episode da gap)
        for t in range(T):
            buf_first[t] = stamp[buf_s[t], buf_a[t]] != ep
            stamp[buf_s[t], buf_a[t]] = ep
        # 3) Tinh return nguoc tu cuoi episode va cap nhat Q.
        #    Bi cat vi max_steps (chua giao het) -> boot_trunc lay max Q(s_T) lam phan con lai
        #    thay vi coi nhu ket thuc voi gia tri 0.
        G = Q[s, _argmax(Q, s)] if (boot_trunc and not term) else 0.0
        for t in range(T - 1, -1, -1):
            G = gamma * G + buf_r[t]
            if buf_first[t] or not first_visit:
                st, at = buf_s[t], buf_a[t]
                N[st, at] += 1
                step = alpha if alpha > 0 else 1.0 / N[st, at]
                Q[st, at] += step * (G - Q[st, at])
        out_ret[ep - ep0], out_len[ep - ep0] = total, T


# ======================================================================
# Ham goi tu Python
# ======================================================================
def greedy_returns(env, Q, starts):
    """Return (khong chiet khau), so buoc, co giao het hay khong cua policy tham lam tu moi trang thai trong starts."""
    NS, R, D = env.tables()
    n = len(starts)
    ret, length, done = np.zeros(n), np.zeros(n, dtype=np.int64), np.zeros(n, dtype=np.bool_)
    _rollouts(Q, NS, R, D, np.asarray(starts, dtype=np.int32), env.max_steps, ret, length, done)
    return ret, length, done


def optimal_returns(env, gamma=0.99):
    """Return toi uu tu moi o xuat phat (Value Iteration) -> moc de tinh do chinh xac. Luu lai tren env."""
    if getattr(env, "_opt_ret", None) is None:
        from planner import value_iteration
        Qv, _ = value_iteration(env, gamma)
        env._opt_ret = greedy_returns(env, Qv, env.start_states("random_cell"))[0]
    return env._opt_ret


def save_q(path, Q):
    """Luu bang Q dang nen (.npz): phan lon hang la trang thai khong dat toi nen nen rat tot."""
    np.savez_compressed(path, Q=Q)


def load_q(path):
    return np.load(path)["Q"]


def run_greedy(env, Q, start=None):
    """Chay 1 episode voi policy tham lam (eps = 0) qua API gymnasium. Tra ve dict ket qua + quy dao."""
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


def _train(chunk, env, eval_env, n_episodes, gamma, seed, eval_every, q_init, on_eval):
    """Khung chung: chay tung doan eval_every episode, sau moi doan danh gia policy tham lam.
    chunk(Q, ep0, ep1, out_ret, out_len) chay cac episode ep0..ep1-1 cua mot thuat toan.
    Lich su luu trung binh moi doan (khong luu tung episode) de file ket qua nho gon."""
    _seed(seed)
    Q = np.full((env.n_states, env.action_space.n), float(q_init))
    s0 = eval_env.start_states("depot")
    cells = eval_env.start_states("random_cell")
    opt = optimal_returns(eval_env, gamma)
    hist = {"eval_ep": [], "eval_return": [], "eval_success": [], "eval_acc": [],
            "train_return": [], "train_len": []}
    out_ret, out_len = np.zeros(eval_every + 1), np.zeros(eval_every + 1, dtype=np.int64)
    # Danh gia sau episode 0, eval_every, 2*eval_every, ... va episode cuoi
    points = sorted(set(range(0, n_episodes, eval_every)) | {n_episodes - 1})
    ep = 0
    for p in points:
        k = p + 1 - ep
        chunk(Q, ep, p + 1, out_ret, out_len)
        ep = p + 1
        ret, _, done = greedy_returns(eval_env, Q, s0)
        cell_ret = greedy_returns(eval_env, Q, cells)[0]
        hist["eval_ep"].append(p)
        hist["eval_return"].append(float(ret[0]))
        hist["eval_success"].append(bool(done[0]))
        hist["eval_acc"].append(float(np.mean(cell_ret >= opt - 1e-9)))  # % o xuat phat di dung lo trinh toi uu
        hist["train_return"].append(float(out_ret[:k].mean()))
        hist["train_len"].append(float(out_len[:k].mean()))
        if on_eval is not None:
            on_eval(p, Q)  # vd: convergence.py do sai so cua Q so voi Q* (khong dung bo sinh ngau nhien)
    return Q, hist


def _td(method, env, eval_env, n_episodes, alpha, alpha_end, gamma, random_first, eps_start, eps_end,
        decay_frac, eps_decay, seed, eval_every, q_init, on_eval):
    NS, R, D = env.tables()
    starts, a1 = env.start_states(), alpha if alpha_end is None else alpha_end

    def chunk(Q, ep0, ep1, out_ret, out_len):
        _td_chunk(Q, NS, R, D, starts, env.max_steps, method, gamma, ep0, ep1, n_episodes, alpha, a1,
                  random_first, eps_start, eps_end, decay_frac, eps_decay == "exp", out_ret, out_len)
    return _train(chunk, env, eval_env, n_episodes, gamma, seed, eval_every, q_init, on_eval)


def q_learning(env, eval_env, n_episodes, alpha=0.5, alpha_end=None, gamma=0.99, random_first=False,
               eps_start=1.0, eps_end=0.05, decay_frac=0.7, eps_decay="linear", seed=0, eval_every=50,
               q_init=0.0, on_eval=None):
    """Q-learning. alpha_end: alpha giam tuyen tinh tu alpha xuong alpha_end (None = giu nguyen).
    eps_decay: "linear" hoac "exp". env.start_mode quyet dinh trang thai bat dau khi huan luyen;
    random_first=True -> hanh dong dau tien cua episode chon ngau nhien (exploring starts cho ca (s, a))."""
    return _td(QL, env, eval_env, n_episodes, alpha, alpha_end, gamma, random_first, eps_start, eps_end,
               decay_frac, eps_decay, seed, eval_every, q_init, on_eval)


def sarsa(env, eval_env, n_episodes, alpha=0.5, alpha_end=None, gamma=0.99, random_first=False,
          eps_start=1.0, eps_end=0.05, decay_frac=0.7, eps_decay="linear", seed=0, eval_every=50,
          q_init=0.0, on_eval=None):
    """SARSA: giong Q-learning nhung muc tieu dung Q(s', a') cua hanh dong se lam that."""
    return _td(SARSA, env, eval_env, n_episodes, alpha, alpha_end, gamma, random_first, eps_start, eps_end,
               decay_frac, eps_decay, seed, eval_every, q_init, on_eval)


def monte_carlo(env, eval_env, n_episodes, gamma=0.99, alpha=None, first_visit=True, boot_trunc=False,
                random_first=False, eps_start=1.0, eps_end=0.05, decay_frac=0.7, eps_decay="linear", seed=0,
                eval_every=50, q_init=0.0, on_eval=None):
    """On-policy MC control.
    alpha=None -> trung binh mau (1/N); alpha=so -> buoc hoc co dinh (theo kip policy dang thay doi).
    first_visit=True -> chi cap nhat lan dau (s,a) xuat hien; False -> every-visit.
    boot_trunc=True -> episode bi cat vi max_steps lay max Q(s_T) lam phan return con lai.
    random_first=True -> hanh dong dau tien cua episode chon ngau nhien (Monte Carlo ES, dung cung
    env.start_mode = "exploring"): moi cap (s, a) deu co co hoi bat dau, nen eps co the giam rat nho."""
    NS, R, D = env.tables()
    starts, T = env.start_states(), env.max_steps
    N = np.zeros((env.n_states, env.action_space.n))
    stamp = np.full(N.shape, -1, dtype=np.int64)
    buf_s, buf_a = np.zeros(T, dtype=np.int32), np.zeros(T, dtype=np.int32)
    buf_r, buf_first = np.zeros(T), np.zeros(T, dtype=np.bool_)

    def chunk(Q, ep0, ep1, out_ret, out_len):
        _mc_chunk(Q, N, stamp, NS, R, D, starts, T, gamma, ep0, ep1, n_episodes, alpha or 0.0,
                  first_visit, boot_trunc, random_first, eps_start, eps_end, decay_frac, eps_decay == "exp",
                  buf_s, buf_a, buf_r, buf_first, out_ret, out_len)
    return _train(chunk, env, eval_env, n_episodes, gamma, seed, eval_every, q_init, on_eval)
