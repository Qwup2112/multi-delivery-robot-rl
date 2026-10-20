"""
delivery_env.py - Moi truong Multi-Delivery Robot (phat trien tu logic Taxi-v3).

Trang thai (state) la mot tuple 5 thanh phan:
    (hang, cot, trang_thai_kien, tau, boost)
  - trang_thai_kien: tuple do dai N, moi phan tu 0 = chua lay, 1 = dang mang, 2 = da giao
  - tau  : so buoc da di, chi dung khi co kien deadline (bi chan tren o T_d)
  - boost: 0 = chua dung o tang tai, 1 = dang co hieu luc, 2 = da dung het
Moi tuple duoc ma hoa thanh MOT so nguyen (encode) de tra bang Q.
"""
import json
from collections import deque

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# ----- Hanh dong -----
UP, DOWN, LEFT, RIGHT, PICKUP, DROP = range(6)
ACTION_NAMES = ["Len", "Xuong", "Trai", "Phai", "Pickup", "Drop"]
MOVES = {UP: (-1, 0), DOWN: (1, 0), LEFT: (0, -1), RIGHT: (0, 1)}

# ----- Trang thai cua mot kien -----
WAITING, CARRYING, DELIVERED = 0, 1, 2
STATUS_NAMES = ["chua lay", "dang mang", "da giao"]


def load_config(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


class MultiDeliveryEnv(gym.Env):
    metadata = {"render_modes": ["ansi"]}

    def __init__(self, config, start_mode="depot", render_mode=None):
        """start_mode: "depot" = luon xuat phat o S (dung de danh gia),
                       "random_cell" = o trong ngau nhien, cac kien deu chua lay (giong Taxi),
                       "exploring" = trang thai ngau nhien bat ky dat toi duoc (exploring starts)."""
        self.cfg = config
        self.name = config["name"]
        self.grid = config["grid"]
        self.H, self.W = len(self.grid), len(self.grid[0])
        self.walls = {(r, c) for r in range(self.H) for c in range(self.W) if self.grid[r][c] == "#"}
        self.depot = self._find("S")
        self.booster = self._find("B")  # None neu map khong co o tang tai

        self.parcels = config["parcels"]
        for p in self.parcels:
            p["pickup"], p["drop"] = tuple(p["pickup"]), tuple(p["drop"])
        self.N = len(self.parcels)
        self.capacity = config["capacity"]
        self.boost_capacity = config.get("boost_capacity", 2)

        rw = config["rewards"]
        self.r_step = rw["step"]
        self.r_illegal = rw["illegal"]
        self.r_late = rw.get("late_penalty", 0)
        self.max_steps = config["max_steps"]
        self.start_mode = start_mode
        self.render_mode = render_mode

        # Kien co deadline (toi da 1 kien)
        self.dl_idx = next((i for i, p in enumerate(self.parcels) if p.get("deadline")), None)
        self.deadline = self.parcels[self.dl_idx]["deadline"] if self.dl_idx is not None else 0

        # Kich thuoc tung thanh phan cua trang thai
        self.n_pos = self.H * self.W
        self.n_codes = 3 ** self.N                                      # moi kien co 3 trang thai
        self.n_tau = self.deadline + 1 if self.dl_idx is not None else 1  # tau = 0..T_d
        self.n_boost = 3 if self.booster else 1
        self.n_states = self.n_pos * self.n_codes * self.n_tau * self.n_boost

        self.observation_space = spaces.Discrete(self.n_states)
        self.action_space = spaces.Discrete(6)

        # O co the xuat phat khi start_mode = "random_cell" (bo tuong va o tang tai)
        self.free_cells = [(r, c) for r in range(self.H) for c in range(self.W)
                           if (r, c) not in self.walls and (r, c) != self.booster]
        self.state = None
        self.t = 0
        self._reach = None   # tap trang thai dat toi (tinh luoi khi can)
        self._tables = None  # bang chuyen trang thai (NS, R, D) cho huan luyen nhanh
        self._pools = {}     # start_mode -> mang trang thai xuat phat

    def _find(self, ch):
        for r, row in enumerate(self.grid):
            if ch in row:
                return (r, row.index(ch))
        return None

    # ------------------------------------------------------------------
    # Ma hoa / giai ma trang thai
    # ------------------------------------------------------------------
    def encode(self, state):
        """tuple -> so nguyen. Trang thai kien doc nhu mot so co so 3."""
        r, c, status, tau, boost = state
        code = 0
        for i, s in enumerate(status):
            code += s * 3 ** i
        idx = r * self.W + c
        idx = idx * self.n_codes + code
        idx = idx * self.n_tau + tau
        idx = idx * self.n_boost + boost
        return idx

    def decode(self, idx):
        """so nguyen -> tuple (lam nguoc lai encode bang divmod)."""
        idx, boost = divmod(idx, self.n_boost)
        idx, tau = divmod(idx, self.n_tau)
        pos, code = divmod(idx, self.n_codes)
        r, c = divmod(pos, self.W)
        status = []
        for _ in range(self.N):
            code, s = divmod(code, 3)
            status.append(s)
        return (r, c, tuple(status), tau, boost)

    # ------------------------------------------------------------------
    # Ham chuyen trang thai (tat dinh)
    # ------------------------------------------------------------------
    def transition(self, state, action):
        """(state, action) -> (next_state, reward, terminated, event)."""
        r, c, status, tau, boost = state
        status = list(status)
        reward = self.r_step
        event = ""

        if action in MOVES:
            dr, dc = MOVES[action]
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.H and 0 <= nc < self.W and (nr, nc) not in self.walls:
                r, c = nr, nc  # dam tuong / ra ngoai -> dung yen
            if (r, c) == self.booster and boost == 0:
                boost = 1
                event = "kich hoat tang tai"

        elif action == PICKUP:
            cap = self.boost_capacity if boost == 1 else self.capacity
            i = self._parcel_here(status, (r, c), WAITING, "pickup")
            if i is not None and status.count(CARRYING) < cap:
                status[i] = CARRYING
                event = f"lay kien {self.parcels[i]['name']}"
            else:
                reward += self.r_illegal
                event = "pickup sai"

        else:  # DROP
            i = self._parcel_here(status, (r, c), CARRYING, "drop")
            if i is not None:
                status[i] = DELIVERED
                reward += self.parcels[i]["reward"]
                event = f"giao kien {self.parcels[i]['name']}"
                if boost == 1 and status.count(CARRYING) == 0:
                    boost = 2  # het mot chuyen -> het hieu luc tang tai
            else:
                reward += self.r_illegal
                event = "drop sai"

        # Dong ho deadline: tau dem so buoc, dung lai o T_d
        if self.dl_idx is not None:
            if status[self.dl_idx] == DELIVERED:
                tau = self.deadline  # da giao -> thoi gian khong con y nghia, gop trang thai
            elif tau < self.deadline:
                tau += 1
                if tau == self.deadline:  # vua het han ma chua giao
                    reward -= self.r_late
                    event = (event + " | tre deadline").strip(" |")

        terminated = all(s == DELIVERED for s in status)
        return (r, c, tuple(status), tau, boost), reward, terminated, event

    def reachable_states(self):
        """BFS tu moi o xuat phat -> tap trang thai co the gap khi chay."""
        if self._reach is None:
            starts = [(r, c, (WAITING,) * self.N, 0, 0) for (r, c) in self.free_cells]
            seen = set(starts)
            queue = deque(starts)
            while queue:
                s = queue.popleft()
                for a in range(6):
                    ns, _, done, _ = self.transition(s, a)
                    if ns not in seen:
                        seen.add(ns)
                        if not done:
                            queue.append(ns)
            self._reach = seen
        return self._reach

    def tables(self):
        """Lap san ham chuyen cho moi (trang thai dat toi, hanh dong) -> 3 mang (n_states x 6):
            NS[s, a] = trang thai ke tiep, R[s, a] = reward, D[s, a] = True neu giao het hang.
        Moi truong tat dinh nen 3 mang nay chua TOAN BO luat choi; agents.py chi tra bang
        thay vi goi transition() moi buoc -> huan luyen nhanh hon rat nhieu (va chay duoc bang numba).
        Trang thai khong dat toi giu NS = chinh no va khong bao gio duoc dung toi."""
        if self._tables is None:
            NS = np.repeat(np.arange(self.n_states, dtype=np.int32), 6).reshape(-1, 6)
            R = np.zeros((self.n_states, 6))
            D = np.zeros((self.n_states, 6), dtype=np.bool_)
            for st in self.reachable_states():
                if all(x == DELIVERED for x in st[2]):
                    continue
                s = self.encode(st)
                for a in range(6):
                    ns, r, done, _ = self.transition(st, a)
                    NS[s, a], R[s, a], D[s, a] = self.encode(ns), r, done
            self._tables = (NS, R, D)
        return self._tables

    def start_states(self, mode=None):
        """Mang cac trang thai (da ma hoa) co the dung de bat dau episode theo start_mode."""
        mode = mode or self.start_mode
        if mode not in self._pools:
            if mode == "exploring":  # moi trang thai dat toi chua giao het hang (exploring starts)
                pool = sorted(s for s in self.reachable_states() if not all(x == DELIVERED for x in s[2]))
            elif mode == "random_cell":
                pool = [(r, c, (WAITING,) * self.N, 0, 0) for (r, c) in self.free_cells]
            else:
                pool = [(*self.depot, (WAITING,) * self.N, 0, 0)]
            self._pools[mode] = np.array([self.encode(s) for s in pool], dtype=np.int32)
        return self._pools[mode]

    def _parcel_here(self, status, pos, need, key):
        for i, p in enumerate(self.parcels):
            if status[i] == need and p[key] == pos:
                return i
        return None

    # ------------------------------------------------------------------
    # API gymnasium
    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        if options and "start" in options:          # chi dinh o xuat phat
            r, c = options["start"]
            self.state = (r, c, (WAITING,) * self.N, 0, 0)
        elif self.start_mode == "exploring":
            pool = self.start_states("exploring")
            self.state = self.decode(int(pool[self.np_random.integers(len(pool))]))
        else:
            if self.start_mode == "random_cell":
                r, c = self.free_cells[self.np_random.integers(len(self.free_cells))]
            else:
                r, c = self.depot
            self.state = (r, c, (WAITING,) * self.N, 0, 0)
        return self.encode(self.state), {}

    def step(self, action):
        self.state, reward, terminated, event = self.transition(self.state, int(action))
        self.t += 1
        truncated = (not terminated) and self.t >= self.max_steps
        return self.encode(self.state), reward, terminated, truncated, {"event": event}

    def render(self):
        """Ve map dang chu. @ = robot, a/b/c = kien cho lay, A/B/C = dich chua giao, * = da giao, + = o tang tai."""
        rr, rc, status, tau, boost = self.state
        lines = []
        for r in range(self.H):
            row = []
            for c in range(self.W):
                ch = "#" if (r, c) in self.walls else "."
                if (r, c) == self.booster:
                    ch = "+" if boost == 0 else "o"  # + = o tang tai chua dung, o = da kich hoat
                for i, p in enumerate(self.parcels):
                    if p["drop"] == (r, c):
                        ch = "*" if status[i] == DELIVERED else p["name"]
                    if p["pickup"] == (r, c) and status[i] == WAITING:
                        ch = p["name"].lower()
                if (r, c) == (rr, rc):
                    ch = "@"
                row.append(ch)
            lines.append(" ".join(row))
        info = ", ".join(f"{p['name']}={STATUS_NAMES[s]}" for p, s in zip(self.parcels, status))
        extra = f"t={self.t}"
        if self.dl_idx is not None:
            extra += f", tau={tau}/{self.deadline}"
        if self.booster:
            extra += f", boost={['chua dung', 'DANG HIEU LUC', 'da dung'][boost]}"
        lines.append(f"Kien: {info} | {extra}")
        return "\n".join(lines)
