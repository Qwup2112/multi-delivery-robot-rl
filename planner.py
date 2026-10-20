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
    S = env.start_states("exploring")  # moi trang thai dat toi chua giao het hang
    NS, R, D = (x[S] for x in env.tables())  # ham chuyen lap san trong delivery_env.py
    D = D.astype(float)

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
    return Q, len(env.reachable_states())
