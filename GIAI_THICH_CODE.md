# Giải thích code – Đề tài 13: Robot giao nhiều kiện hàng

Tài liệu tiếng Việt giải thích **từng file code**: file đó làm gì, các hàm chính hoạt động thế nào, và dữ liệu đi từ đâu tới đâu. Phần toán (phương trình, ý nghĩa thuật toán) đã có trong [`GIAI_THICH.md`](GIAI_THICH.md); ở đây tập trung vào **code**.

Ghi chú: comment và tên biến trong code viết tiếng Việt không dấu. Một số từ hay gặp: *kien* = kiện, *lay* = lấy, *giao* = giao, *sai* = sai, *tre* = trễ, *phat* = phạt, *cu* = cũ (cấu hình cũ), *o tang tai* = ô ×2, *suc chua* = sức chứa.

## Mục lục

1. [Bức tranh tổng thể](#1-bức-tranh-tổng-thể)
2. [`configs/*.json` – định nghĩa màn chơi](#2-configsjson--định-nghĩa-màn-chơi)
3. [`delivery_env.py` – môi trường](#3-delivery_envpy--môi-trường)
4. [`agents.py` – ba thuật toán học](#4-agentspy--ba-thuật-toán-học)
5. [`planner.py` – Value Iteration](#5-plannerpy--value-iteration)
6. [`experiments.py` – chạy toàn bộ thí nghiệm](#6-experimentspy--chạy-toàn-bộ-thí-nghiệm)
7. [`plots.py` – vẽ hình](#7-plotspy--vẽ-hình)
8. [`convergence.py` – đo hội tụ của Q về Q*](#8-convergencepy--đo-hội-tụ-của-q-về-q)
9. [`failure_cases.py` – tìm chỗ robot đi sai](#9-failure_casespy--tìm-chỗ-robot-đi-sai)
10. [`demo.py`, `play.py`, `app.py` – xem và điều khiển robot](#10-demopy-playpy-apppy--xem-và-điều-khiển-robot)
11. [`colab_train.ipynb` và `tools/record_app_demo.sh`](#11-colab_trainipynb-và-toolsrecord_app_demosh)
12. [Thư mục `results/`](#12-thư-mục-results)
13. [Sửa đổi thường gặp](#13-sửa-đổi-thường-gặp)

---

## 1. Bức tranh tổng thể

```
configs/*.json ──► delivery_env.py ──► env.tables(): NS, R, D (luật chơi dạng bảng)
                        │                      │
                        │                      ▼
                        │                agents.py (numba) ──► bảng Q
                        ▼                      │
                   planner.py ──► Q*           ▼
                        │              experiments.py ──► results/ (summary.json, Q_*.npz, *.npz)
                        └──────────────────────┤
                                               ▼
                 plots.py · convergence.py · failure_cases.py ──► figures/, figures_en/
                 demo.py · play.py · app.py ──► xem robot chạy, tự chơi, huấn luyện trực tiếp
```

| File | Dòng | Vai trò |
|---|---|---|
| `configs/*.json` | ~20 mỗi file | 4 bản đồ: lưới, kiện, sức chứa, deadline, ô ×2, phần thưởng, số episode |
| `delivery_env.py` | 281 | Môi trường gymnasium: trạng thái, mã hoá, hàm chuyển, BFS, bảng NS/R/D |
| `agents.py` | 284 | Q-learning, SARSA, Monte Carlo ES (biên dịch bằng numba), chạy chính sách tham lam |
| `planner.py` | 28 | Value Iteration: Q* làm đáp án |
| `experiments.py` | 239 | Chạy mọi thí nghiệm song song, lưu vào `results/` |
| `plots.py` | 628 | Vẽ các hình trong báo cáo |
| `convergence.py` | 211 | Đo sai số \|Q − Q*\| trong lúc học |
| `failure_cases.py` | 558 | Thả robot vào mọi trạng thái, phân loại lỗi, truy mã trạng thái |
| `demo.py` | 111 | Chạy 1 episode và in từng bước |
| `play.py` | 112 | Người tự chơi bằng bàn phím |
| `app.py` | 939 | Ứng dụng đồ hoạ: huấn luyện trực tiếp và điều khiển robot |

**Ý tưởng then chốt về tốc độ:** môi trường tất định, nên toàn bộ luật chơi được tính sẵn một lần thành 3 mảng `NS`, `R`, `D`. Vòng huấn luyện không gọi `env.step()` mà chỉ tra mảng, và được biên dịch bằng numba. Nhờ vậy nhanh hơn Python thường khoảng 150–180 lần, đủ để chạy hàng triệu episode trong vài chục giây.

---

## 2. `configs/*.json` – định nghĩa màn chơi

Ví dụ màn 4 (`configs/s4_booster.json`):

```json
{
  "name": "S4_o_tang_tai",
  "level": 4,
  "grid": ["S...#...", ".##...#.", "....#...", "B#.##.#.",
           "...#....", ".#...##.", ".#.#....", "........"],
  "capacity": 1,
  "boost_capacity": 2,
  "parcels": [
    {"name": "A", "pickup": [0, 7], "drop": [7, 0], "reward": 20},
    {"name": "B", "pickup": [1, 7], "drop": [7, 1], "reward": 20},
    {"name": "C", "pickup": [4, 2], "drop": [0, 3], "reward": 20},
    {"name": "D", "pickup": [7, 7], "drop": [4, 4], "reward": 20, "deadline": 22}
  ],
  "rewards": {"step": -1, "illegal": -10, "late_penalty": 30},
  "max_steps": 300,
  "train": {"episodes": 6000000, "gamma": 0.99, "best": "ql"}
}
```

| Trường | Ý nghĩa |
|---|---|
| `grid` | mỗi chuỗi là một hàng. `S` = ô xuất phát, `#` = tường, `B` = ô ×2, `.` = ô trống |
| `capacity`, `boost_capacity` | số kiện mang được bình thường / khi ô ×2 đang bật |
| `parcels` | mỗi kiện: tên, ô lấy `[hàng, cột]`, ô trả, phần thưởng khi giao, `deadline` (không bắt buộc, tối đa 1 kiện có) |
| `rewards` | `step` mỗi bước, `illegal` khi Pickup/Drop sai, `late_penalty` khi trễ deadline |
| `max_steps` | số bước tối đa mỗi episode |
| `train` | số episode, γ, và thuật toán "tốt nhất" dùng cho demo và các biến thể |

Đổi màn chơi chỉ cần sửa file JSON, **không cần sửa code**.

---

## 3. `delivery_env.py` – môi trường

### Hằng số

```python
UP, DOWN, LEFT, RIGHT, PICKUP, DROP = range(6)          # 6 hành động
MOVES = {UP: (-1, 0), DOWN: (1, 0), LEFT: (0, -1), RIGHT: (0, 1)}
WAITING, CARRYING, DELIVERED = 0, 1, 2                  # trạng thái một kiện
```

### `MultiDeliveryEnv.__init__(config, start_mode)`

Đọc config và tính sẵn kích thước từng thành phần của trạng thái:

- `walls`, `depot` (ô S), `booster` (ô ×2, `None` nếu không có), `free_cells` (ô trống, bỏ tường và ô ×2).
- `dl_idx`, `deadline`: chỉ số và hạn của kiện có deadline.
- `n_pos = H·W`, `n_codes = 3^N`, `n_tau = T_d + 1` (hoặc 1), `n_boost = 3` (hoặc 1), và `n_states` là tích của chúng.
- `start_mode` quyết định episode bắt đầu ở đâu: `"depot"` (luôn ở S, dùng khi đánh giá), `"random_cell"` (ô trống ngẫu nhiên, mọi kiện đang chờ, dùng ở cấu hình cũ), `"exploring"` (trạng thái đạt được bất kỳ, dùng cho exploring starts).

### `encode(state)` và `decode(idx)`

Trạng thái là tuple `(hàng, cột, trạng_thái_kiện, tau, boost)`. `encode` gói nó thành một số nguyên (số hàng trong bảng Q): trạng thái các kiện đọc như số hệ 3, rồi lần lượt nhân và cộng ô, đồng hồ, ô ×2. `decode` làm ngược lại bằng `divmod`. Công thức và ví dụ mã 17419 ở mục 3 của `GIAI_THICH.md`.

### `transition(state, action)` – luật chơi

Hàm thuần (không đổi trạng thái của env), trả về `(trạng_thái_kế, phần_thưởng, giao_hết, sự_kiện)`. Thứ tự xử lý:

1. Mặc định `reward = r_step` (−1).
2. **Di chuyển:** tính ô mới; nếu ra ngoài lưới hoặc là tường thì đứng yên. Bước vào ô ×2 khi `boost == 0` thì `boost = 1` (sự kiện "kich hoat tang tai").
3. **Pickup:** sức chứa là `boost_capacity` nếu `boost == 1`, ngược lại `capacity`. Nếu ở ô có kiện đang chờ và số kiện đang mang chưa đủ sức chứa, kiện chuyển sang "đang mang". Ngược lại cộng thêm `r_illegal` (−10).
4. **Drop:** nếu đang mang một kiện và đứng ở ô trả của nó, kiện chuyển sang "đã giao" và cộng phần thưởng của kiện (+20). Nếu `boost == 1` và robot vừa rảnh tay thì `boost = 2` (hết chuyến ×2). Ngược lại cộng −10.
5. **Đồng hồ deadline** (nếu màn có): kiện deadline đã giao thì `tau = deadline` (gộp trạng thái). Nếu chưa giao thì `tau += 1`, và đúng lúc `tau` chạm `deadline` thì trừ `late_penalty` (một lần duy nhất).
6. `terminated = True` khi mọi kiện đã giao.

### `reachable_states()` – BFS

Duyệt theo chiều rộng từ **mọi ô xuất phát** (ô trống, mọi kiện đang chờ, `tau = 0`, `boost = 0`), thử cả 6 hành động ở mỗi trạng thái, ghi lại mọi trạng thái gặp được. Trạng thái giao hết hàng không duyệt tiếp. Kết quả được lưu lại (`self._reach`) để không tính lại.

### `tables()` – luật chơi dạng bảng

Tạo 3 mảng kích thước `(n_states, 6)`:

- `NS[s, a]`: mã trạng thái kế tiếp.
- `R[s, a]`: phần thưởng.
- `D[s, a]`: `True` nếu bước đó giao hết hàng.

Chỉ điền cho trạng thái đạt được và chưa giao xong; các hàng khác giữ `NS = chính nó` và không bao giờ được dùng. Vì môi trường tất định, 3 mảng này chứa **toàn bộ luật chơi**.

### `start_states(mode)`

Trả về mảng mã các trạng thái có thể bắt đầu episode theo `mode` (`exploring`, `random_cell` hoặc `depot`). Kết quả được lưu lại theo từng mode.

### API gymnasium: `reset`, `step`, `render`

- `reset(options={"start": (hàng, cột)})` cho phép chọn ô xuất phát; nếu không thì theo `start_mode`.
- `step(action)` gọi `transition`, tăng bộ đếm bước, và đặt `truncated = True` khi hết `max_steps` mà chưa giao xong. Trả về `(obs, reward, terminated, truncated, {"event": ...})` đúng chuẩn gymnasium.
- `render()` vẽ bản đồ dạng chữ: `@` robot, `a/b/c/d` kiện đang chờ, `A/B/C/D` ô trả, `*` đã giao, `+` ô ×2, `#` tường.

---

## 4. `agents.py` – ba thuật toán học

### Biên dịch bằng numba (và chạy được khi không có numba)

```python
try:
    from numba import njit
except ImportError:          # không có numba: dùng hàm Python bình thường
    def njit(*args, **kwargs):
        return args[0] if args and callable(args[0]) else (lambda f: f)
```

Các hàm có `@njit(cache=True)` được biên dịch ra mã máy ở lần chạy đầu và lưu cache cho các lần sau.

### Hàm nhỏ dùng chung

| Hàm | Việc làm |
|---|---|
| `_seed(seed)` | đặt seed cho bộ sinh số ngẫu nhiên **bên trong** numba (khác với `np.random` của Python) |
| `_schedule(ep, n, start, end, frac, exponential)` | giá trị của ε hoặc α tại episode `ep`: đi từ `start` tới `end` trong `frac·n` episode đầu, giảm theo hàm mũ hoặc tuyến tính |
| `_argmax(Q, s)` | hành động có Q lớn nhất, hoà thì lấy chỉ số đầu (giống `np.argmax`) |
| `_eps_greedy(Q, s, eps)` | với xác suất ε chọn ngẫu nhiên; ngược lại chọn Q lớn nhất, **hoà thì bốc ngẫu nhiên đều** (giữ ứng viên thứ k với xác suất 1/k) |
| `_rollouts(Q, NS, R, D, starts, ...)` | chạy chính sách tham lam từ nhiều trạng thái, ghi return, số bước, có giao hết không |

### `_td_chunk(...)` – Q-learning và SARSA

Chạy các episode `ep0 … ep1−1`. Mỗi episode:

```python
eps   = _schedule(ep, n_episodes, eps0, eps1, decay_frac, eps_exp)
alpha = _schedule(ep, n_episodes, alpha0, alpha1, 1.0, False)      # α giảm tuyến tính (nếu alpha1 < alpha0)
s = starts[np.random.randint(starts.shape[0])]                     # trạng thái xuất phát ngẫu nhiên
a = np.random.randint(6) if random_first else _eps_greedy(Q, s, eps)
while t < max_steps:
    s2, r, term = NS[s, a], R[s, a], D[s, a]
    if term:               target = r
    elif method == SARSA:  a2 = _eps_greedy(Q, s2, eps); target = r + gamma * Q[s2, a2]
    else:                  target = r + gamma * Q[s2, _argmax(Q, s2)]
    Q[s, a] += alpha * (target - Q[s, a])
    if term: break
    s = s2
    a = a2 if method == SARSA else _eps_greedy(Q, s, eps)
```

Một hàm dùng cho cả hai thuật toán; tham số `method` (`QL` hoặc `SARSA`) chọn công thức target. Khi episode bị cắt vì hết `max_steps`, bước cuối vẫn bootstrap bình thường (bị cắt không phải trạng thái kết thúc).

### `_mc_chunk(...)` – Monte Carlo

Mỗi episode có 3 bước:

1. **Sinh trọn episode** vào các mảng đệm `buf_s`, `buf_a`, `buf_r` (hành động đầu ngẫu nhiên nếu `random_first`).
2. **Đánh dấu lần ghé đầu tiên** của mỗi cặp (s, a) bằng mảng `stamp`: `stamp[s, a]` lưu số episode gần nhất đã gặp cặp đó. Nếu `stamp[s, a] != ep` thì đây là lần đầu trong episode này. Cách này tránh phải xoá một tập hợp sau mỗi episode.
3. **Tính return ngược** `G = gamma * G + r` từ cuối về đầu, và cập nhật Q ở lần ghé đầu tiên với bước học `alpha` (hoặc `1/N` nếu `alpha = 0`, tức lấy trung bình mẫu).

`boot_trunc` (mặc định tắt) quyết định episode bị cắt có lấy `max Q(s_T)` làm phần return còn lại hay coi bằng 0. Cấu hình của đồ án để tắt, nên vòng lặp nhận return rất âm và bị loại.

### `_train(chunk, ...)` – khung huấn luyện và đánh giá

```python
Q = np.full((env.n_states, 6), q_init)
opt = optimal_returns(eval_env, gamma)                # return tối ưu từ mọi ô xuất phát (Value Iteration)
points = range(0, n_episodes, eval_every) ∪ {n_episodes − 1}
for p in points:
    chunk(Q, ep, p + 1, ...)                          # học thêm một đoạn episode
    ret, _, done = greedy_returns(eval_env, Q, s0)    # chạy tham lam từ ô S
    cell_ret = greedy_returns(eval_env, Q, cells)[0]  # và từ mọi ô xuất phát
    hist["eval_acc"].append(mean(cell_ret >= opt − 1e-9))   # độ chính xác
    ...
```

Huấn luyện được chia thành 200 đoạn (`eval_every = n/200`), sau mỗi đoạn đánh giá chính sách tham lam một lần. `hist` lưu trung bình mỗi đoạn chứ không lưu từng episode, nên file kết quả gọn. Tham số `on_eval` là hàm gọi lại sau mỗi lần đánh giá; `convergence.py` dùng nó để đo sai số so với Q*.

### Hàm gọi từ Python

| Hàm | Việc làm |
|---|---|
| `q_learning(...)`, `sarsa(...)` | gọi `_td(...)` với `method` tương ứng; trả về `(Q, hist)` |
| `monte_carlo(...)` | cấp phát `N`, `stamp`, các mảng đệm rồi gọi `_train` với `_mc_chunk` |
| `greedy_returns(env, Q, starts)` | return, số bước, giao hết của chính sách tham lam từ nhiều trạng thái |
| `optimal_returns(env)` | return tối ưu từ mọi ô xuất phát, tính một lần bằng Value Iteration và lưu trên `env` |
| `save_q`, `load_q` | lưu/đọc bảng Q dạng `.npz` nén (phần lớn hàng là trạng thái không đạt tới, toàn số 0, nén rất tốt) |
| `run_greedy(env, Q, start)` | chạy 1 episode qua API gymnasium, trả về return, số bước, quỹ đạo, danh sách sự kiện |

**Tính tái lập:** mọi số ngẫu nhiên trong huấn luyện đều sinh trong các hàm numba sau `_seed(seed)`. Cùng seed và tham số thì ra đúng cùng bảng Q, bất kể chia đoạn thế nào. Nhờ vậy `app.py`, `convergence.py` và `failure_cases.py` huấn luyện lại vẫn khớp kết quả của `experiments.py`.

---

## 5. `planner.py` – Value Iteration

```python
S = env.start_states("exploring")                 # mọi trạng thái đạt được, chưa giao hết
NS, R, D = (x[S] for x in env.tables())           # chỉ lấy các hàng đó
V = np.zeros(env.n_states)
while True:
    Qsa = R + gamma * V[NS] * (1 - D)             # cả ma trận (số trạng thái × 6) một lúc
    new_v = Qsa.max(axis=1)
    delta = np.abs(new_v - V[S]).max()
    V[S] = new_v
    if delta < 1e-8: break
Q = R + gamma * V[NS] * (1 - D)                   # Q*
```

Viết dạng vectơ với numpy nên chạy nhanh. Trả về `(Q*, số trạng thái đạt được)`. Đây chỉ là **đáp án để chấm điểm**, agent không dùng.

---

## 6. `experiments.py` – chạy toàn bộ thí nghiệm

### Các bộ tham số

```python
_ES = {"start_mode": "exploring", "random_first": True}
ALGOS = {   # cấu hình mới
  "ql":    ("Q-learning", q_learning,  dict(_ES, alpha=1.0, eps_end=0.01, decay_frac=0.8, eps_decay="exp")),
  "sarsa": ("SARSA", sarsa,            dict(_ES, alpha=0.5, alpha_end=0.1, eps_end=1e-5, decay_frac=0.5, eps_decay="exp")),
  "mc":    ("Monte Carlo ES", monte_carlo, dict(_ES, alpha=1.0, eps_start=0.0, eps_end=0.0)),
}
_OLD = {"start_mode": "random_cell", "eps_end": 0.05, "decay_frac": 0.7}
BASELINE = { "ql_cu": ..., "sarsa_cu": ..., "mc_cu": ... }     # cấu hình cũ để so sánh
VARIANTS = [ ("S3_phat_tre_0", ...), ("S3_deadline_45", ...), ("S4_khong_tang_tai", ...), ... ]
```

Mỗi biến thể là một hàm nhỏ sửa config (ví dụ `_no_booster` thay `B` bằng `.` trong lưới).

### Các hàm chính

| Hàm | Việc làm |
|---|---|
| `build_run(cfg, key, seed, scale)` | dựng env huấn luyện (theo `start_mode`), env đánh giá (`depot`), số episode và `eval_every`; được `convergence.py` và `failure_cases.py` dùng lại |
| `train_one(job)` | một lần huấn luyện (1 màn × 1 thuật toán × 1 seed), chạy trong process con |
| `optimal_info(cfg)` | Value Iteration + tóm tắt lộ trình tối ưu (thứ tự giao, số bước, có trễ không) |
| `algo_stats(...)` | tổng hợp nhiều seed: return, độ chính xác cuối, episode đầu tiên đạt 100%, thời gian |
| `warmup()` | biên dịch numba một lần ở process chính trước khi chia việc, để các process con không phải biên dịch lại |

### `main()` gồm 3 phần

1. **Kích thước và đáp án tối ưu** của từng màn (in ra `|S|`, số trạng thái đạt được, lộ trình tối ưu).
2. **Phần 1 + 3:** huấn luyện song song bằng `multiprocessing.Pool` mọi tổ hợp màn × thuật toán (mới và cũ) × seed. Việc dài (màn 4, Monte Carlo) được xếp chạy trước để tận dụng CPU.
3. **Phần 2:** các biến thể phần thưởng/môi trường, huấn luyện bằng thuật toán `best`.

Kết quả ghi vào `results/` (xem mục 12). Tham số dòng lệnh: `--quick` (2 seed, 1/20 số episode), `--scale`, `--seeds`, `--no-baseline`, `--workers`.

---

## 7. `plots.py` – vẽ hình

Đọc `results/summary.json`, các file `.npz` và bảng Q, rồi vẽ vào `figures/` (nhãn tiếng Việt) hoặc `figures_en/` (`--lang en`). Hàm `L(vi, en)` chọn chuỗi theo ngôn ngữ.

| Hàm | Hình |
|---|---|
| `draw_grid` | vẽ nền bản đồ: tường, ô S, ô ×2, ô lấy (tròn) và ô trả (vuông); dùng chung cho mọi hình bản đồ |
| `plot_maps` | `maps.png`: 4 bản đồ |
| `plot_state_space` | `state_space.png`: số trạng thái mã hoá và đạt được |
| `_curves` | `learning_curves.png`, `accuracy.png`, `success_rate.png`, `episode_length.png`: đường học trung bình ± độ lệch chuẩn 5 seed |
| `plot_final_bars`, `plot_ablation` | `final_returns.png`, `ablation.png`: so sánh cấu hình cũ và mới |
| `plot_policy` | `policy_S*.png`: chính sách theo từng pha (mũi tên) và lộ trình thật |
| `value_grid`, `draw_heat`, `plot_value_start`, `plot_value_phases` | `value_start.png`, `value_S*.png`: bản đồ màu giá trị V = max Q |
| `plot_variants` | `reward_deadline.png`, `env_booster.png`: các biến thể |

Bảng màu (`ALGO_COLOR`, `PARCEL_COLOR`…) đặt ở đầu file và được `app.py`, `failure_cases.py`, `convergence.py` dùng lại, nên mọi hình cùng một kiểu.

---

## 8. `convergence.py` – đo hội tụ của Q về Q*

Huấn luyện lại (cùng seed và tham số như `experiments.py`) với hàm gọi lại `on_eval`. Sau mỗi lần đánh giá, nó ghi:

| Chỉ số | Ý nghĩa |
|---|---|
| `q_err` | trung bình \|Q(s,a) − Q*(s,a)\| trên mọi trạng thái đạt được |
| `pol_acc` | tỉ lệ trạng thái mà hành động tham lam là một hành động tối ưu theo Q* |
| `pol_route` | như trên nhưng chỉ xét các trạng thái nằm trên lộ trình tối ưu |
| `q_start` | hàng Q(S, ·) tại ô xuất phát |

Hàm `on_eval` không dùng bộ sinh số ngẫu nhiên, nên việc đo không làm lệch quá trình học. Kết quả lưu `results/convergence.npz`; `--plot-only` chỉ vẽ lại.

---

## 9. `failure_cases.py` – tìm chỗ robot đi sai

### Ý tưởng

Thả robot vào **mọi trạng thái đạt được chưa giao xong**, chạy chính sách tham lam và so với Q*. Trạng thái nào thu ít điểm hơn tối ưu là **sai**.

### Các hàm chính

| Hàm | Việc làm |
|---|---|
| `fields(env, idx)` | giải mã **cả mảng** mã trạng thái cùng lúc (`np.divmod`) |
| `rollout(env, Q, starts)` | chạy tham lam từ hàng nghìn trạng thái **song song** qua `NS/R/D`; đếm return, số bước, Pickup/Drop sai (mã kiện không đổi), bị phạt trễ |
| `path(env, Q, s)` | quỹ đạo từ một trạng thái; dừng khi gặp lại trạng thái cũ (chính sách tất định nên robot sẽ lặp mãi) |
| `decisions(env, Q, wrong, starts)` | lần theo lộ trình tới **quyết định sai** đầu tiên: trạng thái đầu tiên mà hành động tham lam không tối ưu theo Q* |
| `analyse(cfg, Q)` | phân loại lỗi, đếm theo pha và theo ô, chọn ví dụ tệ nhất, liệt kê các quyết định sai phổ biến |

### Phân loại lỗi (theo thứ tự ưu tiên)

| Loại | Điều kiện |
|---|---|
| `khong_xong` | hết `max_steps` mà chưa giao hết (lặp vòng) |
| `tre` | bị phạt trễ trong khi lộ trình tối ưu thì không |
| `sai_luat` | có Pickup/Drop sai |
| `di_vong` | chỉ đi nhiều bước hơn tối ưu |

### Bốn nhóm lần chạy (5 seed mỗi nhóm)

`moi` (cấu hình mới; seed 0 dùng bảng Q đã lưu), `cu` (cấu hình cũ), `sarsa_x2` (SARSA gấp đôi episode, màn 3–4), `sarsa_a002` (SARSA với α cuối 0,02, màn 3–4).

Kết quả: `results/failure_cases.json` (tổng hợp) và `results/failure_states.csv` (mỗi trạng thái sai một dòng, kèm mã trạng thái, loại lỗi, số điểm mất, mã quyết định sai). Hình: `failure_rate.png`, `failure_old_routes.png`, `failure_sarsa_S4.png`.

---

## 10. `demo.py`, `play.py`, `app.py` – xem và điều khiển robot

### `demo.py`

`get_q(cfg, algo)` đọc bảng Q từ `results/`; nếu chưa có thì tự huấn luyện nhanh 1 seed bằng `build_run`. `play()` chạy 1 episode tham lam và in bản đồ chữ sau mỗi bước, kèm hành động, phần thưởng và sự kiện. `--gif` lưu thêm ảnh động.

```bash
python demo.py --config configs/s3_deadline.json --algo sarsa
```

### `play.py`

Người chơi thay agent, dùng đúng `env.step()` nên luật và phần thưởng giống hệt lúc agent học. `read_key()` đọc từng phím không cần Enter: trên Windows dùng `msvcrt`, trên Linux/macOS dùng `termios` + `tty`. Nếu không đọc được từng phím (Colab, Git Bash), `key_stream()` cho phép gõ cả chuỗi phím rồi Enter, ví dụ `ddsj`. Hết ván thì so điểm với lộ trình tối ưu.

### `app.py` – kiến trúc

```
┌──────────── process giao diện (Tkinter) ────────────┐      ┌──── process huấn luyện ────┐
│ App: bản đồ, bảng Q/Q*, đường học, nhật ký          │      │ train_worker():            │
│   đọc Q trực tiếp ◄──── bộ nhớ chung (shared_memory) ────► │   gọi _td_chunk/_mc_chunk  │
│   ctrl = mp.Array [STOP, PAUSE, BATCH, DELAY] ─────────────► │   theo từng đoạn episode   │
│   poll() mỗi 50 ms ◄──── hàng đợi tin nhắn (queue) ◄──────── │   gửi ("progress", ...)     │
└──────────────────────────────────────────────────────┘      └────────────────────────────┘
```

- **Huấn luyện chạy trong process riêng** nên giao diện không bị đứng. Bảng Q nằm trong **bộ nhớ chung**, nên giao diện vẽ mũi tên và giá trị từ đúng bảng Q đang học mà không phải sao chép.
- **Mảng điều khiển** `ctrl` gồm 4 ô: dừng, tạm dừng, số episode mỗi lần cập nhật (`BATCH`, 0 = tự động 200 lần), độ trễ (`DELAY`). Giao diện ghi, process huấn luyện đọc.
- `train_worker` gọi đúng các hàm numba của `agents.py`, sau mỗi đoạn đánh giá chính sách tham lam và gửi tin nhắn `progress` (episode, return từ S, độ chính xác, ε, α…). Cùng seed và tham số mặc định thì bảng Q giống hệt `results/Q_*.npz`.
- `presets(key)` lấy tham số mặc định từ chữ ký hàm trong `agents.py` cộng với bộ tham số trong `experiments.ALL`, nên giao diện luôn khớp cấu hình của thí nghiệm.
- `step(a, who)` thực hiện một bước bằng `env.transition`, đánh dấu ⚠ nếu hành động không tối ưu theo Q*, và cuối ván so return với tối ưu. `agent_step()` chọn `argmax Q`; `human(a)` là người chơi; `toggle_auto()` cho agent tự chạy với tốc độ chỉnh được.
- `goto_code()` nhảy tới mã trạng thái bất kỳ (ví dụ 17419); `on_click()` đặt robot vào ô được bấm.

```bash
python app.py --lang en
```

---

## 11. `colab_train.ipynb` và `tools/record_app_demo.sh`

**`colab_train.ipynb`** chạy cả quy trình trên Google Colab: lấy code (từ Google Drive hoặc GitHub), cài `gymnasium numba`, chạy thử `--quick`, chạy đầy đủ (biến `SCALE`, `BASELINE`), vẽ hình hai ngôn ngữ, đo hội tụ, làm ảnh động, in bảng độ chính xác, nén kết quả để tải về. Colab miễn phí thường chỉ có 2 nhân CPU nên chạy đầy đủ mất khoảng 30–45 phút; GPU không giúp gì vì học dạng bảng chạy trên CPU.

**`tools/record_app_demo.sh`** quay video demo (Linux): mở `app.py` trên màn hình ảo 1920×1080 (Xvfb), bấm và gõ phím bằng `xdotool` theo kịch bản cố định (huấn luyện màn 1, chạy robot, huấn luyện màn 4, lớp phủ giá trị, điều khiển tay, gõ mã 17419), và quay bằng `ffmpeg`.

```bash
PYTHON=python3 tools/record_app_demo.sh figures_en/app_demo.mp4 en
```

---

## 12. Thư mục `results/`

| File | Nội dung | Tạo bởi |
|---|---|---|
| `summary.json` | mọi số liệu tổng hợp: kích thước, đáp án tối ưu, độ chính xác, episode đạt 100%, lộ trình học được, biến thể | `experiments.py` |
| `S1_…npz` … `S4_…npz` | đường học của từng thuật toán và seed (return, độ chính xác, độ dài episode) | `experiments.py` |
| `Q_<màn>_<thuật toán>.npz` | bảng Q của seed 0 (cấu hình mới) | `experiments.py` |
| `Q_variant_*.npz`, `config_*.json` | bảng Q và config của các biến thể | `experiments.py` |
| `log.txt` | nhật ký lần chạy đầy đủ | `experiments.py` (qua `tee`) |
| `convergence.npz` | sai số \|Q − Q*\| và các chỉ số hội tụ | `convergence.py` |
| `failure_cases.json`, `failure_states.csv` | phân tích trạng thái sai | `failure_cases.py` |

---

## 13. Sửa đổi thường gặp

**Thêm một màn mới:** tạo `configs/s5_….json` theo mẫu ở mục 2 rồi chạy lại `experiments.py`. Mọi script tự đọc mọi file trong `configs/`. (Riêng `plots.py` có vài hình gọi tên màn cụ thể, như `policy_S4.png`.)

**Đổi phần thưởng:** sửa `rewards` hoặc `reward` của từng kiện trong file JSON. Muốn chạy thành biến thể thì thêm một dòng vào `VARIANTS` trong `experiments.py`, kèm một hàm nhỏ sửa config.

**Đổi tham số thuật toán:** sửa từ điển `ALGOS` trong `experiments.py`. `app.py` tự lấy theo, và cũng cho chỉnh trực tiếp trên giao diện.

**Thêm một thuật toán dạng TD mới** (ví dụ Expected SARSA): thêm một hằng `method` và một nhánh tính `target` trong `_td_chunk`, viết hàm bọc giống `sarsa()`, rồi thêm vào `ALGOS`. Vòng huấn luyện, đánh giá và vẽ hình dùng lại được nguyên.

**Huấn luyện lâu hơn:** `python experiments.py --scale 2` (gấp đôi số episode mọi màn) hoặc sửa `train.episodes` trong config.

**Không có numba:** code vẫn chạy bằng Python thường nhờ hàm `njit` thay thế ở đầu `agents.py`, chỉ chậm hơn rất nhiều. Nên dùng `--quick` để thử.
