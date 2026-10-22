"""
app.py - Ung dung do hoa: huan luyen agent (xem truc tiep tung buoc cua chu trinh hoc) va dieu khien robot.

    python app.py              # giao dien tieng Viet
    python app.py --lang en    # giao dien tieng Anh

Trai : chon map / thuat toan / cau hinh / tham so -> Huan luyen, Tam dung, Dung; so episode moi lan cap nhat
       va do tre (xem cham tung episode). Huan luyen chay trong process rieng bang chinh cac ham numba cua agents.py,
       bang Q nam trong bo nho chung nen ban do va duong hoc cap nhat truc tiep. Cung seed + tham so mac dinh
       -> bang Q giong het results/Q_*.npz cua experiments.py.
Giua : ban do. Lop phu: chinh sach (mui ten), gia tri V, o chon sai so voi Q* (Value Iteration).
       Bam vao o de dat robot vao do; nhap ma trang thai de nhay toi trang thai bat ky.
Phai : trang thai hien tai (ma trang thai, bo gia tri), Q va Q* cua 6 hanh dong, nhat ky tung buoc.
Duoi : duong hoc (do chinh xac, return cua policy tham lam tu o S) theo episode.
Phim : W/A/S/D hoac mui ten = di chuyen, J = Pickup, K = Drop, Space = agent di 1 buoc,
       P = agent tu chay / dung, R = ve o S.
"""
import argparse
import glob
import inspect
import multiprocessing as mp
import os
import queue
import time
import tkinter as tk
from multiprocessing import shared_memory
from tkinter import filedialog, messagebox, ttk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

import agents as AG
import plots as P  # bang mau dung chung voi hinh trong bao cao
from delivery_env import CARRYING, DELIVERED, DROP, MOVES, PICKUP, WAITING, MultiDeliveryEnv, load_config
from experiments import ALL, N_EVAL
from planner import value_iteration

LANG = "vi"


def L(vi, en):
    return en if LANG == "en" else vi


ACTS = [("Up", "Lên"), ("Down", "Xuống"), ("Left", "Trái"), ("Right", "Phải"), ("Pickup", "Pickup"), ("Drop", "Drop")]
KEYMAP = {"w": 0, "Up": 0, "s": 1, "Down": 1, "a": 2, "Left": 2, "d": 3, "Right": 3, "j": PICKUP, "k": DROP}
STOP, PAUSE, BATCH, DELAY = range(4)  # o trong mang dieu khien chung giua GUI va process huan luyen
MAX_BATCH = 100_000  # so episode toi da moi lan goi ham numba (kich thuoc mang ket qua)
BATCHES = [0, 1, 10, 100, 1_000, 10_000, 100_000]  # 0 = tu dong: n_episodes / N_EVAL (giong experiments.py)


def act_name(a):
    return ACTS[a][0] if LANG == "en" else ACTS[a][1]


EVENTS = [("lay kien", "lấy kiện", "picked up"), ("giao kien", "giao kiện", "delivered"),
          ("pickup sai", "Pickup sai (−10)", "wrong Pickup (−10)"), ("drop sai", "Drop sai (−10)", "wrong Drop (−10)"),
          ("tre deadline", "trễ deadline (−30)", "late, deadline missed (−30)"),
          ("kich hoat tang tai", "bật tăng tải", "booster on")]


def event_txt(ev):
    """Su kien cua delivery_env (tieng Viet khong dau) -> chu hien thi."""
    for raw, vi, en in EVENTS:
        ev = ev.replace(raw, L(vi, en))
    return ev


def fmt(x, nd=2):
    return f"{x:.{nd}f}".replace("-", "−")


# ======================================================================
# Process huan luyen
# ======================================================================
def train_worker(cfg, algo, params, seed, n_episodes, shm_name, ctrl, out_q):
    """Huan luyen bang _td_chunk / _mc_chunk cua agents.py, tung doan episode. Q nam trong bo nho chung (GUI doc
    truc tiep). Bo sinh ngau nhien chi duoc dung trong cac doan nen chia doan the nao cung cho ket qua nhu
    q_learning() / sarsa() / monte_carlo()."""
    try:
        env = MultiDeliveryEnv(cfg, params["start_mode"])
        ev = MultiDeliveryEnv(cfg, "depot")
        NS, R, D = env.tables()
        starts, T, gamma = env.start_states(), env.max_steps, cfg["train"]["gamma"]
        shm = shared_memory.SharedMemory(name=shm_name)
        Q = np.ndarray((env.n_states, 6), dtype=np.float64, buffer=shm.buf)
        AG._seed(seed)
        Q[:] = params["q_init"]
        s0, cells = ev.start_states("depot"), ev.start_states("random_cell")
        opt = AG.optimal_returns(ev, gamma)
        out_ret, out_len = np.zeros(MAX_BATCH + 1), np.zeros(MAX_BATCH + 1, dtype=np.int64)
        exp_eps = params["eps_decay"] == "exp"
        if algo == "mc":
            N = np.zeros(Q.shape)
            stamp = np.full(Q.shape, -1, dtype=np.int64)
            bufs = (np.zeros(T, dtype=np.int32), np.zeros(T, dtype=np.int32), np.zeros(T), np.zeros(T, dtype=np.bool_))

            def chunk(e0, e1):
                AG._mc_chunk(Q, N, stamp, NS, R, D, starts, T, gamma, e0, e1, n_episodes, params["alpha"] or 0.0,
                             params["first_visit"], params["boot_trunc"], params["random_first"], params["eps_start"],
                             params["eps_end"], params["decay_frac"], exp_eps, *bufs, out_ret, out_len)
        else:
            method = AG.SARSA if algo == "sarsa" else AG.QL
            a1 = params["alpha"] if params["alpha_end"] is None else params["alpha_end"]

            def chunk(e0, e1):
                AG._td_chunk(Q, NS, R, D, starts, T, method, gamma, e0, e1, n_episodes, params["alpha"], a1,
                             params["random_first"], params["eps_start"], params["eps_end"], params["decay_frac"],
                             exp_eps, out_ret, out_len)

        ep, t0, last = 0, time.time(), 0.0
        while ep < n_episodes and not ctrl[STOP]:
            if ctrl[PAUSE]:
                time.sleep(0.05)
                continue
            batch = int(ctrl[BATCH]) or max(1, n_episodes // N_EVAL)
            k = int(min(n_episodes - ep, batch, MAX_BATCH))
            chunk(ep, ep + k)
            ep += k
            now = time.time()
            if now - last >= 0.04 or ep >= n_episodes or ctrl[DELAY] > 0:  # gioi han so tin nhan gui GUI
                ret, _, done = AG.greedy_returns(ev, Q, s0)
                acc = float(np.mean(AG.greedy_returns(ev, Q, cells)[0] >= opt - 1e-9))
                eps = AG._schedule(ep - 1, n_episodes, params["eps_start"], params["eps_end"], params["decay_frac"],
                                   exp_eps)
                alpha = (None if params["alpha"] is None else
                         AG._schedule(ep - 1, n_episodes, params["alpha"], a1, 1.0, False) if algo != "mc"
                         else params["alpha"])
                out_q.put(("progress", ep, float(ret[0]), bool(done[0]), acc, float(eps), alpha,
                           float(out_ret[:k].mean()), float(out_len[:k].mean()), now - t0))
                last = now
            if ctrl[DELAY] > 0:
                time.sleep(ctrl[DELAY])
        out_q.put(("done" if ep >= n_episodes else "stopped", ep, time.time() - t0))  # shm tu dong khi process ket thuc
    except Exception as e:  # bao loi ve GUI thay vi chet im lang
        out_q.put(("error", repr(e)))


def presets(key):
    """Tham so mac dinh cua 1 thuat toan (ham trong agents.py) + tham so cua cau hinh trong experiments.ALL."""
    _, fn, params = ALL[key]
    p = {k: v.default for k, v in inspect.signature(fn).parameters.items() if v.default is not inspect.Parameter.empty}
    p.update(params)
    p.setdefault("alpha_end", None)
    p.setdefault("first_visit", True)
    p.setdefault("boot_trunc", False)
    return p


# ======================================================================
# Giao dien
# ======================================================================
class App:
    def __init__(self, root):
        self.root = root
        root.title(L("Robot giao nhiều kiện – huấn luyện & điều khiển", "Multi-delivery robot – train & control"))
        root.geometry("1460x900")
        root.minsize(1200, 760)
        self.cfgs = [load_config(p) for p in sorted(glob.glob("configs/*.json"))]
        self.proc = self.shm = self.ctrl = None
        self.msgs = mp.Queue()
        self.Q = None
        self.auto = False
        self.curve = ([], [], [])
        self.last_draw = 0.0
        self.build()
        self.on_map()
        root.bind("<KeyPress>", self.on_key)
        root.protocol("WM_DELETE_WINDOW", self.on_close)
        root.after(50, self.poll)

    # ------------------------------------------------------------------
    # Dung giao dien
    # ------------------------------------------------------------------
    def build(self):
        r = self.root
        r.columnconfigure(1, weight=1)
        r.rowconfigure(0, weight=1)
        left = ttk.Frame(r, padding=8)
        left.grid(row=0, column=0, rowspan=2, sticky="ns")
        mid = ttk.Frame(r, padding=(0, 8))
        mid.grid(row=0, column=1, sticky="nsew")
        right = ttk.Frame(r, padding=8)
        right.grid(row=0, column=2, rowspan=2, sticky="ns")
        bottom = ttk.Frame(r)
        bottom.grid(row=1, column=1, sticky="nsew")

        # ---- Huan luyen ----
        tr = ttk.LabelFrame(left, text=L("Huấn luyện", "Training"), padding=8)
        tr.pack(fill="x")
        self.v_map = tk.StringVar()
        self.v_algo = tk.StringVar(value="ql")
        self.v_preset = tk.StringVar(value="new")
        ttk.Label(tr, text="Map").grid(row=0, column=0, sticky="w")
        self.cb_map = ttk.Combobox(tr, state="readonly", width=34, textvariable=self.v_map,
                                   values=[self.title(c) for c in self.cfgs])
        self.cb_map.grid(row=0, column=1, columnspan=2, sticky="we", pady=2)
        self.cb_map.current(len(self.cfgs) - 1)
        self.cb_map.bind("<<ComboboxSelected>>", lambda e: self.on_map())
        ttk.Label(tr, text=L("Thuật toán", "Algorithm")).grid(row=1, column=0, sticky="w")
        algo_box = ttk.Frame(tr)
        algo_box.grid(row=1, column=1, columnspan=2, sticky="w")
        self.lock = []  # cac o bi khoa trong luc huan luyen
        for k, lab in (("ql", "Q-learning"), ("sarsa", "SARSA"), ("mc", "MC ES")):
            w = ttk.Radiobutton(algo_box, text=lab, value=k, variable=self.v_algo, command=self.on_algo)
            w.pack(side="left", padx=(0, 6))
            self.lock.append(w)
        ttk.Label(tr, text=L("Cấu hình", "Settings")).grid(row=2, column=0, sticky="w")
        pre_box = ttk.Frame(tr)
        pre_box.grid(row=2, column=1, columnspan=2, sticky="w")
        for val, lab in (("new", L("mới (đã cải tiến)", "new (improved)")), ("old", L("cũ", "old"))):
            w = ttk.Radiobutton(pre_box, text=lab, value=val, variable=self.v_preset, command=self.fill_params)
            w.pack(side="left", padx=(0, 6))
            self.lock.append(w)

        self.v_eps_n = tk.StringVar()
        self.v_seed = tk.StringVar(value="0")
        ttk.Label(tr, text="Episodes").grid(row=3, column=0, sticky="w")
        ttk.Entry(tr, textvariable=self.v_eps_n, width=12).grid(row=3, column=1, sticky="w", pady=2)
        sb = ttk.Frame(tr)
        sb.grid(row=3, column=2, sticky="e")
        ttk.Label(sb, text="Seed").pack(side="left")
        ttk.Spinbox(sb, from_=0, to=999, width=5, textvariable=self.v_seed).pack(side="left", padx=(4, 0))

        self.pv = {}
        rows = [("start_mode", L("Xuất phát", "Start states")), ("random_first", L("Hành động đầu ngẫu nhiên",
                                                                                  "Random first action")),
                ("alpha", "α"), ("alpha_end", L("α cuối (trống = giữ)", "final α (blank = keep)")),
                ("eps_start", L("ε đầu", "initial ε")), ("eps_end", L("ε cuối", "final ε")),
                ("decay_frac", L("Giảm ε trong (tỉ lệ)", "ε decays over (fraction)")),
                ("eps_decay", L("Kiểu giảm ε", "ε decay")), ("q_init", L("Q ban đầu", "initial Q"))]
        pf = ttk.Frame(tr)
        pf.grid(row=4, column=0, columnspan=3, sticky="we", pady=(4, 2))
        for i, (k, lab) in enumerate(rows):
            ttk.Label(pf, text=lab).grid(row=i, column=0, sticky="w")
            if k == "random_first":
                v = tk.BooleanVar()
                w = ttk.Checkbutton(pf, variable=v)
            elif k in ("start_mode", "eps_decay"):
                v = tk.StringVar()
                w = ttk.Combobox(pf, state="readonly", width=12, textvariable=v,
                                 values=["exploring", "random_cell", "depot"] if k == "start_mode" else ["exp", "linear"])
            else:
                v = tk.StringVar()
                w = ttk.Entry(pf, textvariable=v, width=12)
            w.grid(row=i, column=1, sticky="w", padx=(8, 0), pady=1)
            self.pv[k] = (v, w)
            self.lock.append(w)

        bf = ttk.Frame(tr)
        bf.grid(row=5, column=0, columnspan=3, sticky="we", pady=(6, 2))
        self.b_train = ttk.Button(bf, text=L("▶ Huấn luyện", "▶ Train"), command=self.start_train)
        self.b_train.pack(side="left", expand=True, fill="x")
        self.b_pause = ttk.Button(bf, text=L("Tạm dừng", "Pause"), command=self.toggle_pause, state="disabled")
        self.b_pause.pack(side="left", expand=True, fill="x", padx=4)
        self.b_stop = ttk.Button(bf, text=L("■ Dừng", "■ Stop"), command=self.stop_train, state="disabled")
        self.b_stop.pack(side="left", expand=True, fill="x")

        spd = ttk.Frame(tr)
        spd.grid(row=6, column=0, columnspan=3, sticky="we")
        ttk.Label(spd, text=L("Episode / lần cập nhật", "Episodes per update")).grid(row=0, column=0, sticky="w")
        self.v_batch = tk.StringVar()
        names = [L("tự động (200 lần)", "auto (200 updates)")] + [f"{b:,}" for b in BATCHES[1:]]
        self.cb_batch = ttk.Combobox(spd, state="readonly", width=16, values=names, textvariable=self.v_batch)
        self.cb_batch.current(0)
        self.cb_batch.grid(row=0, column=1, sticky="w", padx=(8, 0))
        self.cb_batch.bind("<<ComboboxSelected>>", lambda e: self.push_ctrl())
        ttk.Label(spd, text=L("Chậm lại (ms)", "Slow down (ms)")).grid(row=1, column=0, sticky="w")
        self.v_delay = tk.IntVar(value=0)
        ttk.Scale(spd, from_=0, to=1000, variable=self.v_delay,
                  command=lambda v: self.push_ctrl()).grid(row=1, column=1, sticky="we", padx=(8, 0))
        self.pb = ttk.Progressbar(tr, maximum=1.0)
        self.pb.grid(row=7, column=0, columnspan=3, sticky="we", pady=(6, 2))
        self.l_prog = ttk.Label(tr, text="", width=46)
        self.l_prog.grid(row=8, column=0, columnspan=3, sticky="w")
        self.l_sched = ttk.Label(tr, text="")
        self.l_sched.grid(row=9, column=0, columnspan=3, sticky="w")
        self.l_eval = ttk.Label(tr, text="")
        self.l_eval.grid(row=10, column=0, columnspan=3, sticky="w")
        qf = ttk.Frame(tr)
        qf.grid(row=11, column=0, columnspan=3, sticky="we", pady=(6, 0))
        ttk.Button(qf, text=L("Nạp Q (experiments)", "Load Q (experiments)"),
                   command=self.load_saved).pack(side="left", expand=True, fill="x")
        ttk.Button(qf, text=L("Nạp…", "Load…"), command=self.load_file).pack(side="left", padx=4)
        ttk.Button(qf, text=L("Lưu…", "Save…"), command=self.save_file).pack(side="left")
        self.l_qsrc = ttk.Label(tr, text="", foreground=P.INK2, wraplength=330)
        self.l_qsrc.grid(row=12, column=0, columnspan=3, sticky="w", pady=(2, 0))

        # ---- Dieu khien robot ----
        ct = ttk.LabelFrame(left, text=L("Điều khiển robot", "Robot control"), padding=8)
        ct.pack(fill="x", pady=(8, 0))
        b1 = ttk.Frame(ct)
        b1.pack(fill="x")
        ttk.Button(b1, text=L("Về ô S (R)", "Back to S (R)"), command=self.reset_robot).pack(side="left", expand=True,
                                                                                         fill="x")
        ttk.Button(b1, text=L("Ô ngẫu nhiên", "Random cell"), command=self.random_cell).pack(side="left", expand=True,
                                                                                          fill="x", padx=4)
        b2 = ttk.Frame(ct)
        b2.pack(fill="x", pady=4)
        ttk.Button(b2, text=L("Agent 1 bước (Space)", "Agent step (Space)"),
                   command=self.agent_step).pack(side="left", expand=True, fill="x")
        self.b_auto = ttk.Button(b2, text=L("Agent tự chạy (P)", "Agent run (P)"), command=self.toggle_auto)
        self.b_auto.pack(side="left", expand=True, fill="x", padx=4)
        sp = ttk.Frame(ct)
        sp.pack(fill="x")
        ttk.Label(sp, text=L("Tốc độ (ms / bước)", "Speed (ms / step)")).pack(side="left")
        self.v_speed = tk.IntVar(value=250)
        ttk.Scale(sp, from_=20, to=1000, variable=self.v_speed).pack(side="left", expand=True, fill="x", padx=(8, 0))
        pad = ttk.Frame(ct)
        pad.pack(pady=(6, 2))
        for txt, a, rr, cc in (("↑ W", 0, 0, 1), ("← A", 2, 1, 0), ("↓ S", 1, 1, 1), ("→ D", 3, 1, 2),
                               ("Pickup J", PICKUP, 0, 3), ("Drop K", DROP, 1, 3)):
            ttk.Button(pad, text=txt, width=9, command=lambda a=a: self.human(a)).grid(row=rr, column=cc, padx=2, pady=2)
        cs = ttk.Frame(ct)
        cs.pack(fill="x", pady=(4, 0))
        ttk.Label(cs, text=L("Mã trạng thái", "State code")).pack(side="left")
        self.v_code = tk.StringVar()
        e = ttk.Entry(cs, textvariable=self.v_code, width=10)
        e.pack(side="left", padx=4)
        e.bind("<Return>", lambda ev: self.goto_code())
        ttk.Button(cs, text=L("Đi tới", "Go"), command=self.goto_code).pack(side="left")
        ttk.Label(ct, text=L("Bấm vào một ô trên bản đồ để đặt robot vào đó.", "Click a cell to put the robot there."),
                  foreground=P.INK2).pack(anchor="w", pady=(4, 0))

        # ---- Ban do ----
        ov = ttk.Frame(mid)
        ov.pack(fill="x")
        ttk.Label(ov, text=L("Lớp phủ:", "Overlay:")).pack(side="left")
        self.v_overlay = tk.StringVar(value="policy")
        for k, lab in (("none", L("không", "none")), ("policy", L("chính sách", "policy")),
                       ("value", L("giá trị V", "value V")), ("wrong", L("sai so với Q*", "wrong vs Q*"))):
            ttk.Radiobutton(ov, text=lab, value=k, variable=self.v_overlay,
                            command=self.draw_map).pack(side="left", padx=4)
        self.v_live = tk.BooleanVar(value=True)
        ttk.Checkbutton(ov, text=L("cập nhật khi huấn luyện", "live while training"),
                        variable=self.v_live).pack(side="right")
        self.cv = tk.Canvas(mid, bg="white", highlightthickness=0)
        self.cv.pack(fill="both", expand=True)
        self.cv.bind("<Configure>", lambda e: self.draw_map())
        self.cv.bind("<Button-1>", self.on_click)
        for cb in (self.cb_map, self.cb_batch):  # chon xong -> tra phim cho ban do (W/A/S/D...)
            cb.bind("<<ComboboxSelected>>", lambda e: self.cv.focus_set(), add="+")

        # ---- Trang thai / Q ----
        st = ttk.LabelFrame(right, text=L("Trạng thái hiện tại", "Current state"), padding=8)
        st.pack(fill="x")
        self.l_code = ttk.Label(st, text="", font=("Segoe UI", 13, "bold"))
        self.l_code.pack(anchor="w")
        self.l_tuple = ttk.Label(st, text="", font=("Consolas", 11))
        self.l_tuple.pack(anchor="w")
        self.l_info = ttk.Label(st, text="", wraplength=330, justify="left")
        self.l_info.pack(anchor="w", pady=(4, 0))
        qt = ttk.LabelFrame(right, text=L("Q của 6 hành động", "Q of the 6 actions"), padding=8)
        qt.pack(fill="x", pady=(8, 0))
        self.tv = ttk.Treeview(qt, columns=("a", "q", "qs", "m"), show="headings", height=6)
        for c, txt, w in (("a", L("Hành động", "Action"), 80), ("q", L("Q (đã học)", "Q (learned)"), 90),
                          ("qs", "Q*", 80), ("m", "", 60)):
            self.tv.heading(c, text=txt)
            self.tv.column(c, width=w, anchor="e" if c in ("q", "qs") else "w")
        self.tv.tag_configure("greedy", background="#dbe9fb")
        self.tv.tag_configure("bad", background="#fbdcdc")
        self.tv.pack(fill="x")
        self.l_verdict = ttk.Label(qt, text="", wraplength=330)
        self.l_verdict.pack(anchor="w", pady=(4, 0))
        lg = ttk.LabelFrame(right, text=L("Nhật ký", "Log"), padding=8)
        lg.pack(fill="both", expand=True, pady=(8, 0))
        self.log = tk.Text(lg, width=44, height=16, font=("Consolas", 9), wrap="word", state="disabled")
        self.log.pack(fill="both", expand=True)

        # ---- Duong hoc ----
        self.fig = Figure(figsize=(8, 2.3), dpi=96)
        self.ax_acc = self.fig.add_subplot(1, 2, 1)
        self.ax_ret = self.fig.add_subplot(1, 2, 2)
        self.fig.subplots_adjust(left=0.07, right=0.98, bottom=0.2, top=0.86, wspace=0.25)
        self.fc = FigureCanvasTkAgg(self.fig, master=bottom)
        self.fc.get_tk_widget().pack(fill="both", expand=True)

    # ------------------------------------------------------------------
    # Map / thuat toan / tham so
    # ------------------------------------------------------------------
    def on_map(self):
        if self.proc is not None:
            return
        self.cfg = self.cfgs[self.cb_map.current()]
        self.env = MultiDeliveryEnv(self.cfg, "depot")
        self.reach = self.env.reachable_states()
        self.Qstar, _ = value_iteration(self.env, gamma=self.cfg["train"]["gamma"])
        s0 = self.env.encode((*self.env.depot, (WAITING,) * self.env.N, 0, 0))
        self.opt_S = self.rollout(self.Qstar, s0)
        self.v_eps_n.set(f"{self.cfg['train']['episodes']:,}".replace(",", " "))
        self.fill_params()
        self.load_saved(quiet=True)
        self.reset_robot()
        self.clear_curve()

    def on_algo(self):
        self.fill_params()
        if self.proc is None:
            self.load_saved(quiet=True)
            self.refresh()

    def key(self):
        return self.v_algo.get() + ("_cu" if self.v_preset.get() == "old" else "")

    def fill_params(self):
        p = presets(self.key())
        for k, (v, w) in self.pv.items():
            val = p.get(k)
            if isinstance(v, tk.BooleanVar):
                v.set(bool(val))
            else:
                v.set("" if val is None else str(val))
        mc = self.v_algo.get() == "mc"
        self.pv["alpha_end"][1].configure(state="disabled" if mc else "normal")
        self.preset = p

    def read_params(self):
        p = dict(self.preset)
        for k, (v, _) in self.pv.items():
            if isinstance(v, tk.BooleanVar):
                p[k] = bool(v.get())
            elif k in ("start_mode", "eps_decay"):
                p[k] = v.get()
            else:
                txt = v.get().strip()
                p[k] = None if txt == "" and k in ("alpha", "alpha_end") else float(txt)
        if self.v_algo.get() != "mc" and p["alpha"] is None:
            raise ValueError(L("α không được để trống với Q-learning / SARSA", "α is required for Q-learning / SARSA"))
        return p

    # ------------------------------------------------------------------
    # Huan luyen
    # ------------------------------------------------------------------
    def start_train(self):
        try:
            params = self.read_params()
            n = int(self.v_eps_n.get().replace(" ", "").replace(",", ""))
            seed = int(self.v_seed.get())
        except ValueError as e:
            messagebox.showerror(L("Tham số sai", "Bad parameter"), str(e))
            return
        self.stop_auto()
        shape = (self.env.n_states, 6)
        self.release_shm()
        self.shm = shared_memory.SharedMemory(create=True, size=int(np.prod(shape)) * 8)
        self.Q = np.ndarray(shape, dtype=np.float64, buffer=self.shm.buf)
        self.Q[:] = params["q_init"]
        self.ctrl = mp.Array("d", 4)
        self.push_ctrl()
        algo = self.v_algo.get()
        self.n_train = n
        self.proc = mp.Process(target=train_worker, daemon=True,
                               args=(self.cfg, algo, params, seed, n, self.shm.name, self.ctrl, self.msgs))
        self.proc.start()
        self.train_t0 = time.time()
        self.clear_curve()
        self.set_training(True)
        name = {"ql": "Q-learning", "sarsa": "SARSA", "mc": "Monte Carlo ES"}[algo]
        self.run_name = (f"{name} ({L('cũ' if self.v_preset.get() == 'old' else 'mới', self.v_preset.get())}), "
                         f"seed {seed}")
        self.l_qsrc.configure(text=L(f"Q: đang huấn luyện {self.run_name}", f"Q: training {self.run_name}"))
        self.refresh()
        self.write(L(f"▶ Huấn luyện {name} trên {self.short()}: {n:,} episode, seed {seed}",
                     f"▶ Training {name} on {self.short()}: {n:,} episodes, seed {seed}"))

    def push_ctrl(self):
        if self.ctrl is not None:
            self.ctrl[BATCH] = BATCHES[max(0, self.cb_batch.current())]
            self.ctrl[DELAY] = self.v_delay.get() / 1000.0

    def toggle_pause(self):
        if self.ctrl is None:
            return
        self.ctrl[PAUSE] = 0.0 if self.ctrl[PAUSE] else 1.0
        self.b_pause.configure(text=L("Tiếp tục", "Resume") if self.ctrl[PAUSE] else L("Tạm dừng", "Pause"))

    def stop_train(self):
        if self.ctrl is not None:
            self.ctrl[PAUSE] = 0.0
            self.ctrl[STOP] = 1.0

    def set_training(self, on):
        for w in self.lock:
            w.configure(state="disabled" if on else ("readonly" if isinstance(w, ttk.Combobox) else "normal"))
        if not on and self.v_algo.get() == "mc":
            self.pv["alpha_end"][1].configure(state="disabled")
        self.b_train.configure(state="disabled" if on else "normal")
        self.b_pause.configure(state="normal" if on else "disabled", text=L("Tạm dừng", "Pause"))
        self.b_stop.configure(state="normal" if on else "disabled")
        self.cb_map.configure(state="disabled" if on else "readonly")

    def finish_train(self, msg):
        if self.proc is None:
            return
        self.proc.join(timeout=2)
        self.proc = None
        self.Q = np.array(self.Q)  # chep ra khoi bo nho chung roi giai phong
        self.release_shm()
        self.ctrl = None
        self.set_training(False)
        self.l_qsrc.configure(text=L(f"Q: vừa huấn luyện {self.run_name} (chưa lưu)",
                                     f"Q: just trained {self.run_name} (not saved)"))
        self.write(msg)
        self.refresh()

    def release_shm(self):
        if self.shm is not None:
            if self.Q is not None and self.Q.base is not None and self.proc is None:
                self.Q = np.array(self.Q)
            try:
                self.shm.close()
                self.shm.unlink()
            except (BufferError, FileNotFoundError):
                pass
            self.shm = None

    def poll(self):
        got = False
        try:
            while True:
                m = self.msgs.get_nowait()
                if m[0] == "progress":
                    _, ep, ret, done, acc, eps, alpha, tr_ret, tr_len, dt = m
                    self.curve[0].append(ep)
                    self.curve[1].append(100 * acc)
                    self.curve[2].append(ret)
                    self.pb["value"] = ep / self.n_train
                    self.l_prog.configure(text=L(f"Episode {ep:,} / {self.n_train:,} · {ep / max(dt, 1e-9):,.0f} ep/s",
                                                 f"Episode {ep:,} / {self.n_train:,} · {ep / max(dt, 1e-9):,.0f} ep/s"))
                    a_txt = "1/N" if alpha is None else f"{alpha:.3g}"
                    self.l_sched.configure(text=f"ε = {eps:.4g} · α = {a_txt} · "
                                                + L(f"episode huấn luyện: return {fmt(tr_ret, 1)}, {tr_len:.0f} bước",
                                                    f"training episodes: return {fmt(tr_ret, 1)}, {tr_len:.0f} steps"))
                    self.l_eval.configure(text=L(f"Độ chính xác {100 * acc:.1f}% · return từ S {fmt(ret, 0)} "
                                                 f"(tối ưu {fmt(self.opt_S[0], 0)})",
                                                 f"Accuracy {100 * acc:.1f}% · return from S {fmt(ret, 0)} "
                                                 f"(optimal {fmt(self.opt_S[0], 0)})"))
                    got = True
                elif m[0] in ("done", "stopped"):
                    _, ep, dt = m
                    self.draw_curve()
                    self.finish_train(L(f"■ {'Xong' if m[0] == 'done' else 'Đã dừng'} sau {ep:,} episode, {dt:.1f} s",
                                        f"■ {'Finished' if m[0] == 'done' else 'Stopped'} after {ep:,} episodes, "
                                        f"{dt:.1f} s"))
                elif m[0] == "error":
                    self.finish_train("Error: " + m[1])
        except queue.Empty:
            pass
        if got and time.time() - self.last_draw > 0.25:
            self.last_draw = time.time()
            self.draw_curve()
            if self.v_live.get():
                self.refresh()
        self.root.after(50, self.poll)

    # ------------------------------------------------------------------
    # Nap / luu bang Q
    # ------------------------------------------------------------------
    def load_saved(self, quiet=False):
        path = os.path.join("results", f"Q_{self.cfg['name']}_{self.v_algo.get()}.npz")
        if os.path.exists(path):
            self.Q = AG.load_q(path)
            self.l_qsrc.configure(text=L(f"Q: {path} (experiments.py, cấu hình mới, seed 0)",
                                         f"Q: {path} (experiments.py, new settings, seed 0)"))
        else:
            self.Q = None
            self.l_qsrc.configure(text=L("Q: chưa có – hãy huấn luyện", "Q: none yet – train first"))
            if not quiet:
                messagebox.showinfo("Q", L(f"Chưa có {path}", f"{path} not found"))
        if not quiet:
            self.refresh()

    def load_file(self):
        path = filedialog.askopenfilename(initialdir="results", filetypes=[("Q table", "*.npz")])
        if path:
            Q = AG.load_q(path)
            if Q.shape != (self.env.n_states, 6):
                messagebox.showerror("Q", L("Bảng Q không khớp với map đang chọn", "Q table does not match this map"))
                return
            self.Q = Q
            self.l_qsrc.configure(text=f"Q: {os.path.basename(path)}")
            self.refresh()

    def save_file(self):
        if self.Q is None:
            return
        name = f"app_Q_{self.cfg['name']}_{self.key()}_seed{self.v_seed.get()}.npz"
        path = filedialog.asksaveasfilename(initialdir="results", initialfile=name, defaultextension=".npz")
        if path:
            AG.save_q(path, np.array(self.Q))
            self.write(L(f"Đã lưu {path}", f"Saved {path}"))

    # ------------------------------------------------------------------
    # Robot
    # ------------------------------------------------------------------
    def set_state(self, state, why):
        self.state = state
        self.start = state
        self.t, self.total, self.done = 0, 0.0, False
        self.trail = [state[:2]]
        self.write(f"— {why}: s = {self.env.encode(state)} {self.tuple_txt(state)}")
        self.refresh()

    def reset_robot(self):
        self.stop_auto()
        self.set_state((*self.env.depot, (WAITING,) * self.env.N, 0, 0), L("Về ô S", "Back to S"))

    def random_cell(self):
        self.stop_auto()
        r, c = self.env.free_cells[np.random.randint(len(self.env.free_cells))]
        self.set_state((r, c, (WAITING,) * self.env.N, 0, 0), L("Ô ngẫu nhiên", "Random cell"))

    def goto_code(self):
        try:
            code = int(self.v_code.get())
            st = self.env.decode(code)
        except (ValueError, IndexError):
            st = None
        if st is None or st not in self.reach or all(x == DELIVERED for x in st[2]):
            messagebox.showwarning(L("Mã trạng thái", "State code"),
                                   L("Mã này không phải trạng thái đạt tới được (chưa giao hết) của map này.",
                                     "Not a reachable, unfinished state of this map."))
            return
        self.stop_auto()
        self.set_state(st, L(f"Đi tới mã {code}", f"Go to code {code}"))

    def on_click(self, e):
        self.cv.focus_set()
        cell = self.cell_at(e.x, e.y)
        if cell is None or cell in self.env.walls:
            return
        _, _, status, tau, boost = self.state
        st = (*cell, status, tau, boost)
        if self.done or st not in self.reach:  # vd: dung tren o x2 khi chua kich hoat -> khong the xay ra
            st = (*cell, (WAITING,) * self.env.N, 0, 0)
            if st not in self.reach:
                return
        self.stop_auto()
        self.set_state(st, L(f"Đặt robot ở ô {cell}", f"Robot put at cell {cell}"))

    def step(self, a, who):
        if self.done:
            self.write(L("Hết ván – bấm R hoặc chọn trạng thái khác.", "Episode over – press R or pick a state."))
            return
        s = self.env.encode(self.state)
        optimal = self.Qstar[s, a] >= self.Qstar[s].max() - 1e-6
        ns, r, term, event = self.env.transition(self.state, a)
        self.state = ns
        self.t += 1
        self.total += r
        self.trail.append(ns[:2])
        trunc = (not term) and self.t >= self.env.max_steps
        mark = "" if optimal else L("  ⚠ không tối ưu", "  ⚠ not optimal")
        self.write(f"{self.t:>3} {who:<6} {act_name(a):<7} r={fmt(r, 0):>4}  {event_txt(event)}{mark}")
        if term or trunc:
            self.done = True
            self.stop_auto()
            opt = self.rollout(self.Qstar, self.env.encode(self.start))
            res = L("giao đủ hàng" if term else f"hết {self.env.max_steps} bước", "all delivered" if term
                    else f"{self.env.max_steps} steps used up")
            self.write(L(f"KẾT QUẢ: {res}, return {fmt(self.total, 0)} trong {self.t} bước · tối ưu từ điểm xuất "
                         f"phát: {fmt(opt[0], 0)} trong {opt[1]} bước",
                         f"RESULT: {res}, return {fmt(self.total, 0)} in {self.t} steps · optimal from the start: "
                         f"{fmt(opt[0], 0)} in {opt[1]} steps"))
        self.refresh()

    def human(self, a):
        self.stop_auto()
        self.step(a, L("Bạn", "You"))

    def agent_step(self):
        if self.Q is None:
            self.write(L("Chưa có bảng Q – hãy huấn luyện hoặc nạp Q.", "No Q table – train or load one."))
            return
        self.step(int(np.argmax(self.Q[self.env.encode(self.state)])), "Agent")

    def toggle_auto(self):
        if self.auto:
            self.stop_auto()
        else:
            if self.done:
                self.reset_robot()
            self.auto = True
            self.b_auto.configure(text=L("Dừng agent (P)", "Stop agent (P)"))
            self.auto_tick()

    def auto_tick(self):
        if not self.auto:
            return
        self.agent_step()
        if self.done or self.Q is None:
            self.stop_auto()
        else:
            self.root.after(int(self.v_speed.get()), self.auto_tick)

    def stop_auto(self):
        self.auto = False
        if hasattr(self, "b_auto"):
            self.b_auto.configure(text=L("Agent tự chạy (P)", "Agent run (P)"))

    def on_key(self, e):
        if isinstance(e.widget, (tk.Entry, ttk.Entry, ttk.Spinbox, ttk.Combobox)):
            return
        k = e.keysym if e.keysym in KEYMAP else e.keysym.lower()
        if k in KEYMAP:
            self.human(KEYMAP[k])
        elif k == "space":
            self.stop_auto()
            self.agent_step()
        elif k == "p":
            self.toggle_auto()
        elif k == "r":
            self.reset_robot()

    def rollout(self, Q, s):
        """Return / so buoc cua policy tham lam tu ma trang thai s (dung cho Q*)."""
        st, total, t = self.env.decode(s), 0.0, 0
        while t < self.env.max_steps:
            st, r, term, _ = self.env.transition(st, int(np.argmax(Q[self.env.encode(st)])))
            total += r
            t += 1
            if term:
                break
        return total, t

    # ------------------------------------------------------------------
    # Hien thi
    # ------------------------------------------------------------------
    def short(self):
        return self.cfg["name"].split("_")[0]

    @staticmethod
    def title(cfg):
        k = cfg["name"].split("_")[0]
        return L(P.TITLE.get(k, cfg["title"]), P.TITLE_EN.get(k, cfg["title"]))

    def tuple_txt(self, st):
        r, c, status, tau, boost = st
        return "(" + ", ".join(str(v) for v in (r, c, *status, tau, boost)) + ")"

    def write(self, line):
        self.log.configure(state="normal")
        self.log.insert("end", line + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def refresh(self):
        self.draw_map()
        self.show_state()

    def show_state(self):
        st = self.state
        s = self.env.encode(st)
        r, c, status, tau, boost = st
        self.l_code.configure(text=L(f"Mã trạng thái: {s}", f"State code: {s}"))
        self.l_tuple.configure(text="s = " + self.tuple_txt(st))
        names = {WAITING: L("chờ", "waiting"), CARRYING: L("đang cầm", "carried"), DELIVERED: L("đã giao", "delivered")}
        parts = [f"{p['name']} {names[x]}" for p, x in zip(self.env.parcels, status)]
        info = L(f"Ô ({r},{c}) · bước {self.t}/{self.env.max_steps} · return {fmt(self.total, 0)}",
                 f"Cell ({r},{c}) · step {self.t}/{self.env.max_steps} · return {fmt(self.total, 0)}")
        info += "\n" + ", ".join(parts)
        if self.env.dl_idx is not None:
            info += f"\nτ = {tau}/{self.env.deadline}"
        if self.env.booster:
            info += " · " + L("tăng tải: ", "booster: ") + [L("chưa bật", "off"), L("ĐANG BẬT", "ON"),
                                                           L("đã hết", "used")][boost]
        self.l_info.configure(text=info)
        self.tv.delete(*self.tv.get_children())
        finished = all(x == DELIVERED for x in status)
        qs = self.Qstar[s]
        best = set(np.flatnonzero(qs >= qs.max() - 1e-6)) if not finished else set()
        g = int(np.argmax(self.Q[s])) if self.Q is not None else None
        for a in range(6):
            q = "–" if self.Q is None else fmt(self.Q[s, a])
            mark = ("▶ " if a == g else "") + ("★" if a in best else "")
            tag = ("bad",) if (a == g and best and a not in best) else ("greedy",) if a == g else ()
            self.tv.insert("", "end", values=(act_name(a), q, "–" if finished else fmt(qs[a]), mark), tags=tag)
        if finished:
            verdict = L("Đã giao hết hàng.", "All parcels delivered.")
        elif g is None:
            verdict = L("Chưa có bảng Q.", "No Q table yet.")
        elif g in best:
            verdict = L(f"✓ Agent chọn {act_name(g)} – tối ưu (▶ agent, ★ tối ưu theo Q*)",
                        f"✓ Agent takes {act_name(g)} – optimal (▶ agent, ★ optimal under Q*)")
        else:
            verdict = L(f"⚠ Rẽ sai: agent chọn {act_name(g)}, tối ưu là "
                        f"{'/'.join(act_name(a) for a in sorted(best))} (chênh Q* {fmt(qs.max() - qs[g])})",
                        f"⚠ Wrong decision: agent takes {act_name(g)}, optimal is "
                        f"{'/'.join(act_name(a) for a in sorted(best))} (Q* gap {fmt(qs.max() - qs[g])})")
        self.l_verdict.configure(text=verdict, foreground=P.INK if g is None or g in best or finished else "#c0392b")

    def geometry(self):
        w, h = max(self.cv.winfo_width(), 50), max(self.cv.winfo_height(), 50)
        H, W = self.env.H, self.env.W
        size = min((w - 20) / W, (h - 20) / H)
        return size, (w - size * W) / 2, (h - size * H) / 2

    def cell_at(self, x, y):
        size, ox, oy = self.geometry()
        c, r = int((x - ox) // size), int((y - oy) // size)
        return (r, c) if 0 <= r < self.env.H and 0 <= c < self.env.W else None

    def draw_map(self):
        if not hasattr(self, "state"):
            return
        cv, env = self.cv, self.env
        cv.delete("all")
        size, ox, oy = self.geometry()
        box = lambda r, c, m=0: (ox + c * size + m, oy + r * size + m, ox + (c + 1) * size - m, oy + (r + 1) * size - m)  # noqa: E731
        ctr = lambda r, c: (ox + (c + 0.5) * size, oy + (r + 0.5) * size)  # noqa: E731
        _, _, status, tau, boost = self.state
        overlay = self.v_overlay.get()
        cells = {}
        for r in range(env.H):
            for c in range(env.W):
                st = (r, c, status, tau, boost)
                if (r, c) not in env.walls and st in self.reach and not all(x == DELIVERED for x in status):
                    cells[(r, c)] = env.encode(st)
        vals = {k: self.Q[s].max() for k, s in cells.items()} if self.Q is not None else {}
        lo, hi = (min(vals.values()), max(vals.values())) if vals else (0, 1)
        for r in range(env.H):
            for c in range(env.W):
                fill = "#f6f5f2"
                if (r, c) in env.walls:
                    fill = P.WALL
                elif (r, c) == env.booster:
                    fill = "#f6dca0" if boost == 0 else "#fbf1db"
                if overlay == "value" and (r, c) in vals:
                    x = (vals[(r, c)] - lo) / (hi - lo) if hi > lo else 0.5
                    rgba = P.HEAT_CMAP(x)
                    fill = "#%02x%02x%02x" % tuple(int(255 * v) for v in rgba[:3])
                elif overlay == "wrong" and (r, c) in cells and self.Q is not None:
                    s = cells[(r, c)]
                    ok = self.Qstar[s, int(np.argmax(self.Q[s]))] >= self.Qstar[s].max() - 1e-6
                    fill = "#dff3e4" if ok else "#f8c9c9"
                cv.create_rectangle(*box(r, c), fill=fill, outline="#d9d8d3")
                if overlay == "value" and (r, c) in vals:
                    x = (vals[(r, c)] - lo) / (hi - lo) if hi > lo else 0.5
                    cv.create_text(ox + (c + 0.5) * size, oy + (r + 0.82) * size, text=fmt(vals[(r, c)], 1),
                                   fill="white" if x > P.HEAT_DARK else P.INK, font=("Segoe UI", max(7, int(size / 9))))
        if env.booster:
            cv.create_text(*ctr(*env.booster), text="×2", font=("Segoe UI", int(size / 4), "bold"), fill=P.INK)
        dr, dc = env.depot
        cv.create_text(ox + dc * size + 4, oy + dr * size + 3, text="S", anchor="nw",
                       font=("Segoe UI", max(8, int(size / 7)), "bold"), fill=P.INK2)
        for i, p in enumerate(env.parcels):
            col = P.PARCEL_COLOR[p["name"]]
            qr, qc = p["drop"]
            x0, y0, x1, y1 = box(qr, qc, size * 0.2)
            if status[i] == DELIVERED:
                cv.create_rectangle(x0, y0, x1, y1, outline=col, width=3)
                cv.create_text(*ctr(qr, qc), text="✓", fill=col, font=("Segoe UI", int(size / 4), "bold"))
            else:
                cv.create_rectangle(x0, y0, x1, y1, fill=col, outline="white", width=2)
                cv.create_text(*ctr(qr, qc), text=p["name"], fill="white", font=("Segoe UI", int(size / 4), "bold"))
            if status[i] == WAITING:
                pr, pc = p["pickup"]
                cv.create_oval(*box(pr, pc, size * 0.22), fill=col, outline="white", width=2)
                cv.create_text(*ctr(pr, pc), text=p["name"].lower(), fill="white",
                               font=("Segoe UI", int(size / 4), "bold"))
        if overlay == "policy" and self.Q is not None:
            for (r, c), s in cells.items():
                a = int(np.argmax(self.Q[s]))
                x, y = ctr(r, c)
                if a in MOVES:
                    dr_, dc_ = MOVES[a]
                    cv.create_line(x - dc_ * size * 0.22, y - dr_ * size * 0.22, x + dc_ * size * 0.3,
                                   y + dr_ * size * 0.3, arrow="last", width=2, fill="#8f8e88",
                                   arrowshape=(size * 0.14, size * 0.17, size * 0.06))
                else:
                    cv.create_text(x + size * 0.3, y + size * 0.3, text="P" if a == PICKUP else "D",
                                   font=("Segoe UI", max(8, int(size / 7)), "bold"), fill=P.INK2)
        if len(self.trail) > 1:
            pts = [v for rc in self.trail[-300:] for v in ctr(*rc)]
            cv.create_line(*pts, fill="#3d3c39", width=2, dash=(4, 3))
        rr, rc = self.state[:2]
        held = [env.parcels[i]["name"] for i, x in enumerate(status) if x == CARRYING]
        col = P.PARCEL_COLOR[held[0]] if len(held) == 1 else (P.INK if held else P.EMPTY_PATH)
        cv.create_oval(*box(rr, rc, size * 0.3), fill=col, outline="white", width=3)
        cv.create_text(*ctr(rr, rc), text="+".join(held) if held else "R", fill="white",
                       font=("Segoe UI", max(8, int(size / 6)), "bold"))

    def clear_curve(self):
        self.curve = ([], [], [])
        self.draw_curve()

    def draw_curve(self):
        col = P.ALGO_COLOR[self.v_algo.get()] if self.v_preset.get() == "new" else P.OLD_COLOR
        for ax in (self.ax_acc, self.ax_ret):
            ax.clear()
            ax.grid(True, color=P.GRIDC)
            P.kilo(ax)
        ep, acc, ret = self.curve
        self.ax_acc.plot(ep, acc, color=col, lw=2)
        self.ax_acc.set_ylim(-3, 103)
        self.ax_acc.set_title(L("Độ chính xác (% ô xuất phát đi tối ưu)", "Accuracy (% start cells optimal)"),
                              fontsize=10, loc="left")
        self.ax_ret.plot(ep, ret, color=col, lw=2)
        self.ax_ret.axhline(self.opt_S[0], color=P.INK2, lw=1, ls="--")
        self.ax_ret.set_ylim(min([self.opt_S[0] - 5] + ret[-200:]) - 5, self.opt_S[0] + 8)
        self.ax_ret.set_title(L("Return từ ô S (nét đứt = tối ưu)", "Return from S (dashed = optimal)"),
                              fontsize=10, loc="left")
        if not ep:
            self.ax_acc.text(0.5, 0.5, L("Bấm ▶ Huấn luyện để vẽ đường học", "Press ▶ Train to draw the curves"),
                             transform=self.ax_acc.transAxes, ha="center", color=P.INK2)
        for ax in (self.ax_acc, self.ax_ret):
            ax.set_xlim(0, max(ep[-1] if ep else 1, 1) if self.proc is None else self.n_train)
        self.fc.draw_idle()

    def on_close(self):
        if self.proc is not None:
            self.stop_train()
            self.proc.join(timeout=2)
            if self.proc.is_alive():
                self.proc.terminate()
            self.proc = None
        self.release_shm()
        self.root.destroy()


def main():
    global LANG
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", choices=["vi", "en"], default="vi", help="ngon ngu giao dien")
    LANG = ap.parse_args().lang
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    root = tk.Tk()
    try:
        ttk.Style().theme_use("vista")
    except tk.TclError:
        pass
    App(root)
    root.mainloop()


if __name__ == "__main__":
    mp.freeze_support()
    main()
