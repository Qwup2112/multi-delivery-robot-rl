"""
planner.py - Tinh policy toi uu bang Value Iteration (dung lam MOC so sanh).

Vi moi truong tat dinh va nho, ta liet ke duoc moi trang thai dat toi (BFS)
roi lap phuong trinh Bellman toi uu:
    V(s) = max_a [ r(s,a) + gamma * V(s') * (1 - done) ]
Day KHONG phai thuat toan hoc; chi de biet "return tot nhat co the" cua moi map.
"""
import numpy as np


def value_iteration(env, gamma=0.99, theta=1e-8):
    """Tra ve (Q_opt, so trang thai dat toi)."""
    states = list(env.reachable_states())
    nonterm = [s for s in states if not all(x == 2 for x in s[2])]
    S = np.array([env.encode(s) for s in nonterm])
    NS = np.zeros((len(nonterm), 6), dtype=np.int64)
    R = np.zeros((len(nonterm), 6))
    D = np.zeros((len(nonterm), 6))
    for k, s in enumerate(nonterm):
        for a in range(6):
            ns, r, done, _ = env.transition(s, a)
            NS[k, a], R[k, a], D[k, a] = env.encode(ns), r, done

    V = np.zeros(env.n_states)
    while True:
        Qsa = R + gamma * V[NS] * (1 - D)
        new_v = Qsa.max(axis=1)
        delta = np.abs(new_v - V[S]).max()
        V[S] = new_v
        if delta < theta:
            break
    Q = np.zeros((env.n_states, 6))
    Q[S] = R + gamma * V[NS] * (1 - D)
    return Q, len(states)
