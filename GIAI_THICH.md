# Giải thích phương trình và thuật toán – Đề tài 13: Robot giao nhiều kiện hàng

Tài liệu ôn tập bằng tiếng Việt cho buổi bảo vệ. Mỗi công thức có ví dụ bằng số lấy từ chính đồ án (đã đối chiếu với code). Phần mô tả thuật toán bám đúng cách cài đặt trong `agents.py`, `experiments.py` và `planner.py`.

Sinh viên: Nguyễn Quang Trung · MSSV 22011211 · K16 AI&RB · Giảng viên: ThS. Vũ Hoàng Diệu

## Mục lục

1. [Bài toán, phần thưởng và return](#1-bài-toán-phần-thưởng-và-return)
2. [Trạng thái và tính Markov](#2-trạng-thái-và-tính-markov)
3. [Mã hoá trạng thái (state code) và kích thước không gian trạng thái](#3-mã-hoá-trạng-thái-state-code-và-kích-thước-không-gian-trạng-thái)
4. [Value Iteration: đáp án tối ưu để chấm điểm](#4-value-iteration-đáp-án-tối-ưu-để-chấm-điểm)
5. [Khung chung của ba thuật toán học](#5-khung-chung-của-ba-thuật-toán-học)
6. [Q-learning](#6-q-learning-học-sau-mỗi-bước-off-policy)
7. [SARSA](#7-sarsa-học-sau-mỗi-bước-on-policy)
8. [Monte Carlo ES](#8-monte-carlo-es-học-sau-cả-episode)
9. [Lịch giảm ε và α](#9-lịch-giảm-ε-và-α)
10. [Đánh giá: độ chính xác và trạng thái sai](#10-đánh-giá-độ-chính-xác-và-trạng-thái-sai)
11. [So sánh ba thuật toán, cấu hình cũ và cấu hình mới](#11-so-sánh-ba-thuật-toán-cấu-hình-cũ-và-cấu-hình-mới)
12. [Câu hỏi thường gặp khi bảo vệ](#12-câu-hỏi-thường-gặp-khi-bảo-vệ)
13. [Bảng ký hiệu: toán ↔ code](#13-bảng-ký-hiệu-toán--code)

---

## 1. Bài toán, phần thưởng và return

Robot đi trên lưới ô vuông có tường, xuất phát ở ô S. Mỗi kiện hàng có một ô lấy (hình tròn, chữ thường) và một ô trả (hình vuông, chữ hoa). Robot có 6 hành động: Lên, Xuống, Trái, Phải, Pickup (lấy), Drop (trả). Robot **tự học thứ tự lấy và giao** để được nhiều điểm nhất.

Mỗi bước robot nhận phần thưởng $r$:

| Sự kiện | $r$ |
|---|---|
| Mỗi bước đi (kể cả đâm vào tường, robot đứng yên) | $-1$ |
| Giao đúng một kiện | $+20$ (cộng với $-1$ của bước đó, nên log ghi $r = 19$) |
| Pickup/Drop sai chỗ hoặc quá sức chứa | thêm $-10$ |
| Kiện có deadline chưa giao khi đồng hồ $\tau$ chạm $T_d$ (chỉ tính 1 lần) | $-30$ |

**Return** là tổng phần thưởng của cả episode:

$$R = \sum_{t} r_t$$

Ví dụ màn 4: lộ trình tối ưu đi 67 bước và giao 4 kiện, nên $R = 4 \cdot 20 - 67 = 13$. Tương tự màn 1: $2 \cdot 20 - 22 = 18$; màn 2: $3 \cdot 20 - 36 = 24$; màn 3: $3 \cdot 20 - 44 = 16$.

**Ý nghĩa:** phạt $-1$ mỗi bước khiến đường **ngắn hơn** thì return **cao hơn**. "Tối đa hoá return" vì vậy chính là "tìm thứ tự giao ngắn nhất mà không trễ deadline và không làm sai".

Episode **kết thúc** (terminated) khi mọi kiện đã giao, và **bị cắt** (truncated) khi hết `max_steps` (100 / 150 / 200 / 300 bước cho màn 1–4). Bị cắt là giới hạn thời gian chứ không phải trạng thái kết thúc thật (xem câu hỏi 12.7).

### Bốn màn chơi

| Màn | Cấp độ | Lưới | Kiện | Sức chứa | Cơ chế mới | Lộ trình tối ưu |
|---|---|---|---|---|---|---|
| S1 | 1 · Dễ | 5×5 | 2 | 2 | mang được 2 kiện cùng lúc | A → B: 22 bước, return 18 |
| S2 | 2 · Trung bình | 6×6 | 3 | 1 | mỗi lần chỉ mang 1, thứ tự quan trọng | B → A → C: 36 bước, return 24 |
| S3 | 3 · Khó | 7×7 | 3 | 1 | kiện C phải giao trong 20 bước | C → B → A: 44 bước, return 16 |
| S4 | 4 · Rất khó | 8×8 | 4 | 1 (2 khi bật ô ×2) | ô ×2 + kiện D deadline 22 bước | D → C → B → A: 67 bước, return 13 |

Màn 3 và 4 có **đánh đổi** mà một quy tắc "lấy kiện gần nhất" sẽ bỏ lỡ:
- Màn 3: không có deadline thì B → A → C chỉ 40 bước, nhưng C sẽ trễ, nên robot phải đi lấy C trước (giao ở bước 20, vừa đúng hạn).
- Màn 4: robot giao D trước (bước 22, vừa đúng hạn), giao C, rồi đi vòng qua ô ×2 (bước 39) để mang A và B cùng lúc. Bỏ ô ×2 thì phải 75 bước, nên đi vòng tiết kiệm được 8 bước.

---

## 2. Trạng thái và tính Markov

$$s = (\text{row},\ \text{col},\ \sigma_1, \dots, \sigma_N,\ \tau,\ b)$$

- $(\text{row}, \text{col})$: ô robot đang đứng.
- $\sigma_i \in \{0, 1, 2\}$: kiện $i$ đang **chờ** / **đang mang** / **đã giao**. Số kiện đang mang chính là số $\sigma_i = 1$, nên không cần thêm biến "tải hiện tại".
- $\tau \in \{0, \dots, T_d\}$: đồng hồ đếm bước tới deadline (chỉ màn 3 và 4).
- $b \in \{0, 1, 2\}$: ô ×2 **chưa dùng** / **đang bật** / **đã dùng hết** (chỉ màn 4). Chuyến ×2 bắt đầu khi robot bước vào ô ×2 và kết thúc lần đầu tiên robot rảnh tay sau một lần Drop; ô chỉ dùng được 1 lần mỗi episode.

**Tính Markov** nghĩa là tương lai chỉ phụ thuộc trạng thái hiện tại, không phụ thuộc quá khứ. Các thuật toán RL dạng bảng cần điều này để một ô Q(s, a) có một giá trị duy nhất.

- **Thiếu $\tau$:** hai lần robot đứng cùng một ô, cùng trạng thái kiện, có thể có tương lai khác nhau (một lần còn kịp deadline, một lần đã trễ). Cùng một "trạng thái" mà hai giá trị là vi phạm Markov.
- **Thiếu $b$:** kết quả của Pickup (mang được 1 hay 2 kiện) phụ thuộc việc robot *đã từng* đi qua ô ×2 hay chưa, tức là phụ thuộc lịch sử.

**Gộp trạng thái:** khi kiện có deadline đã giao xong, $\tau$ được gán bằng $T_d$ luôn, vì thời gian không còn ảnh hưởng tới phần thưởng. Nhờ vậy các trạng thái chỉ khác nhau ở $\tau$ được gộp lại, bảng Q nhỏ hơn và học nhanh hơn.

---

## 3. Mã hoá trạng thái (state code) và kích thước không gian trạng thái

Bảng Q là một ma trận, nên mỗi trạng thái cần một **chỉ số hàng** (một số nguyên). Hàm `encode()` trong `delivery_env.py` làm việc này:

$$\text{code} = \sum_{i=1}^{N} \sigma_i \cdot 3^{\,i-1} \qquad \text{(trạng thái các kiện đọc như một số hệ 3)}$$

$$\text{idx} = \Big( \big( (\text{row} \cdot W + \text{col}) \cdot 3^N + \text{code} \big) \cdot T + \tau \Big) \cdot B + b$$

với $T = T_d + 1$ nếu màn có deadline ($T = 1$ nếu không) và $B = 3$ nếu có ô ×2 ($B = 1$ nếu không). Hàm `decode()` làm ngược lại bằng các phép `divmod`.

**Ví dụ:** mã **17419** ở màn 4 ($W = 8$, $N = 4$, $T = 23$, $B = 3$). Trạng thái $(0, 3,\ \sigma = 0,0,1,0,\ \tau = 10,\ b = 1)$:

- $\text{code} = 1 \cdot 3^2 = 9$
- $\text{idx} = \big(((0 \cdot 8 + 3) \cdot 81 + 9) \cdot 23 + 10\big) \cdot 3 + 1 = (252 \cdot 23 + 10) \cdot 3 + 1 = 5806 \cdot 3 + 1 = 17419$ ✓

Đây là ô trả hàng của C, robot đang mang C, $\tau = 10$, ô ×2 đang bật. Đây đúng là chỗ SARSA ra quyết định sai (mục 10). Trong `app.py`, gõ mã này vào ô "Mã trạng thái" là robot nhảy tới đúng trạng thái đó.

**Kích thước không gian trạng thái:**

$$|S| = H \cdot W \cdot 3^N \cdot T \cdot B$$

| Màn | Công thức | $\lvert S \rvert$ mã hoá | Đạt được (BFS) |
|---|---|---|---|
| S1 | $5 \cdot 5 \cdot 3^2$ | 225 | 186 |
| S2 | $6 \cdot 6 \cdot 3^3$ | 972 | 535 |
| S3 | $7 \cdot 7 \cdot 3^3 \cdot 21$ | 27.783 | 4.027 |
| S4 | $8 \cdot 8 \cdot 3^4 \cdot 23 \cdot 3$ | 357.696 | 15.612 |

Số trạng thái **đạt được** ít hơn nhiều: ô tường không thể đứng, sức chứa 1 loại bỏ trường hợp mang 2 kiện, còn $\tau$ và $b$ chỉ nhận những giá trị thật sự xảy ra. Tập đạt được tìm bằng duyệt theo chiều rộng (BFS) từ ô S.

---

## 4. Value Iteration: đáp án tối ưu để chấm điểm

Phương trình Bellman tối ưu (`planner.py`):

$$V(s) \leftarrow \max_a \big[\, r(s,a) + \gamma \cdot V(s') \cdot (1 - \text{done}) \,\big]$$

Giá trị của một trạng thái = phần thưởng của hành động tốt nhất + giá trị (đã chiết khấu) của trạng thái kế tiếp. Nếu bước đó giao hết hàng (done = 1) thì không còn tương lai. Lặp trên mọi trạng thái đạt được cho tới khi $V$ thay đổi ít hơn $10^{-8}$. Từ đó:

$$Q^*(s,a) = r(s,a) + \gamma \cdot V(s') \cdot (1 - \text{done})$$

Môi trường **tất định và nhỏ**, nên Value Iteration cho **đáp án chính xác**. Nó **không phải thuật toán học**: agent không dùng nó. Nó chỉ là "đáp án" để tính độ chính xác, sai số $|Q - Q^*|$ và tìm trạng thái sai.

**Hệ số chiết khấu $\gamma = 0{,}99$:** phần thưởng nhận sau $k$ bước chỉ còn đáng $0{,}99^k$. Vì vậy giá trị học được tại S nhỏ hơn return thật: màn 1–4 lần lượt là 13,7 / 17,8 / 8,6 / 2,5, trong khi return tối ưu là 18 / 24 / 16 / 13. Các phần thưởng $+20$ đến muộn bị chiết khấu nhiều hơn.

---

## 5. Khung chung của ba thuật toán học

Cả ba học một **bảng Q** kích thước (số trạng thái × 6), khởi tạo bằng 0. $Q(s,a)$ là ước lượng *"đang ở $s$, làm $a$, rồi chơi tốt về sau thì được bao nhiêu điểm (có chiết khấu)"*. Học xong, robot đi theo **chính sách tham lam**:

$$\pi(s) = \arg\max_a Q(s, a)$$

Cả ba dùng cùng một dạng cập nhật và chỉ khác nhau ở **target**:

$$Q(s,a) \leftarrow Q(s,a) + \alpha \cdot \big[\, \text{target} - Q(s,a) \,\big]$$

$\alpha$ là bước học: $Q$ dịch về phía target một đoạn bằng $\alpha$. Với $\alpha = 1$ thì $Q$ được gán bằng target luôn.

Bốn thành phần dùng chung:

1. **Bảng chuyển trạng thái tính sẵn.** Môi trường tất định nên `env.tables()` tính trước 3 mảng: `NS[s,a]` là trạng thái kế, `R[s,a]` là phần thưởng, `D[s,a]` cho biết bước đó có giao hết hàng không. Vòng huấn luyện chỉ tra mảng và được biên dịch bằng **numba**. Ví dụ màn 2 với 20.000 episode: Q-learning giảm từ 10,7 s xuống 0,06 s, Monte Carlo từ 15,4 s xuống 0,10 s, tức nhanh hơn khoảng 150–180 lần.
2. **ε-greedy** (`_eps_greedy`): với xác suất $\varepsilon$ chọn hành động ngẫu nhiên; ngược lại chọn hành động có $Q$ lớn nhất, hoà nhau thì bốc ngẫu nhiên đều giữa các hành động bằng nhau.
3. **Exploring starts:** mỗi episode huấn luyện bắt đầu ở **một trạng thái đạt được ngẫu nhiên** (bất kỳ vị trí, trạng thái kiện, $\tau$, $b$), và **hành động đầu tiên cũng ngẫu nhiên**. Nhờ vậy mọi cặp $(s, a)$ đều được cập nhật trực tiếp, kể cả trạng thái kiểu "đang mang C, $\tau = 15$" mà robot xuất phát từ $\tau = 0$ hiếm khi gặp. Khi đánh giá thì robot vẫn xuất phát bình thường (mọi kiện đang chờ, $\tau = 0$).
4. **Đánh giá định kỳ:** cứ $n/200$ episode, code chạy chính sách tham lam ($\varepsilon = 0$) từ ô S và từ mọi ô xuất phát, so với Value Iteration để vẽ đường học và tính độ chính xác.

---

## 6. Q-learning: học sau mỗi bước, off-policy

Hàm `_td_chunk` với `method = QL`:

```
s ← trạng thái đạt được ngẫu nhiên;  a ← hành động ngẫu nhiên     (exploring starts)
lặp tối đa max_steps bước:
    s', r, done ← NS[s,a], R[s,a], D[s,a]
    nếu done:  target = r                                        (hết hàng, không còn tương lai)
    ngược lại: target = r + γ · max_a' Q(s', a')                 (hành động TỐT NHẤT ở s')
    Q(s,a) ← Q(s,a) + α · (target − Q(s,a))
    nếu done: dừng
    s ← s';  a ← ε-greedy(s')                                    (bước sau vẫn có thể khám phá)
```

$$\text{target} = \begin{cases} r, & \text{nếu } s' \text{ là trạng thái kết thúc} \\ r + \gamma \max_{a'} Q(s', a'), & \text{ngược lại} \end{cases}$$

**Ví dụ giá trị lan ngược** (với $\alpha = 1$, $Q$ ban đầu bằng 0):

- Robot đứng ở ô trả hàng, mang kiện cuối cùng, bấm Drop: $r = -1 + 20 = 19$, hết hàng, nên target = 19 và $Q(\text{ô đó}, \text{Drop}) = 19$.
- Ở ô ngay trước đó, bước sang ô trả hàng: target $= -1 + 0{,}99 \cdot 19 = 17{,}81$.
- Thêm một ô nữa: $-1 + 0{,}99 \cdot 17{,}81 = 16{,}63$, và cứ thế lan dần về các ô xa hơn.

Đó là cách giá trị của một lần giao hàng **lan ngược** trên bản đồ, từ đích về điểm xuất phát. Bản đồ màu (heatmap) giá trị trong báo cáo cho thấy đúng điều này: ô càng gần mục tiêu càng đậm.

**Vì sao gọi là off-policy:** target dùng $\max$, tức hành động tốt nhất ở $s'$, *bất kể* bước sau robot có thật sự làm hành động đó hay không. Vì vậy Q-learning học thẳng $Q^*$ (giá trị của chính sách tối ưu) ngay cả khi đang đi ngẫu nhiên để khám phá.

**Tham số và lý do:**

| Tham số | Giá trị | Lý do |
|---|---|---|
| $\alpha$ | 1 | môi trường tất định nên target không có nhiễu; gán thẳng là đúng ngay, $\alpha$ nhỏ hơn chỉ làm chậm |
| $\varepsilon$ | 1 → 0,01, giảm theo hàm mũ trong 80% số episode đầu | khám phá nhiều lúc đầu, về sau gần như chỉ khai thác |

**Kết quả:** nhanh nhất và ổn định nhất. Màn 4 cần trung vị 1,14 triệu episode để đạt 100%; sau đó không bao giờ tụt xuống; $|Q - Q^*| = 0$ khi kết thúc.

---

## 7. SARSA: học sau mỗi bước, on-policy

Vòng lặp giống hệt Q-learning (`_td_chunk` với `method = SARSA`), chỉ khác target:

```
    a' ← ε-greedy(s')                       (chọn TRƯỚC hành động sẽ làm thật ở s')
    target = r + γ · Q(s', a')
    Q(s,a) ← Q(s,a) + α · (target − Q(s,a))
    s, a ← s', a'                           (bước sau làm đúng a' đó)
```

$$\text{target} = r + \gamma \, Q(s', a'), \qquad a' \sim \varepsilon\text{-greedy}(s')$$

Tên gọi đến từ bộ 5 giá trị dùng để cập nhật: **S**, **A**, **R**, **S**′, **A**′.

**Khác biệt cốt lõi:** nếu ở $s'$ robot đang khám phá và bốc phải một hành động tệ (ví dụ Pickup sai chỗ, $Q = -25$), thì target cũng tệ theo. SARSA học **giá trị của chính chính sách đang chạy**, kể cả các bước ngẫu nhiên. Đó là ý nghĩa của **on-policy**.

**Hệ quả và cách xử lý:**

- SARSA chỉ hội tụ về $Q^*$ khi $\varepsilon \to 0$ (điều kiện GLIE: khám phá vô hạn lần nhưng tiến dần về tham lam). Vì vậy $\varepsilon$ giảm tới $10^{-5}$, nhanh, trong 50% số episode đầu.
- Target có nhiễu vì $a'$ đôi khi ngẫu nhiên. Do đó $\alpha$ giảm tuyến tính 0,5 → 0,1 trong suốt quá trình huấn luyện để lấy trung bình bớt nhiễu.
- Kết quả: SARSA cần khoảng **3 lần** số episode so với Q-learning ở màn 4 (3,57 triệu so với 1,14 triệu), và độ chính xác còn tụt dưới 100% ở 9–19% số lần đánh giá sau khi đã đạt 100%.
- Sau huấn luyện vẫn còn sai số nhỏ ($|Q - Q^*|$ từ 0,001 đến 0,047 trung bình). Ở chỗ hai hành động gần bằng nhau trong $Q^*$, sai số này đủ để đảo thứ tự hai hành động. Đó là lý do SARSA còn đi vòng ở 1,8% trạng thái của màn 4 (mục 10).

**Ưu điểm thường thấy của SARSA** là đi an toàn hơn khi đang khám phá, vì nó tính cả rủi ro của bước ngẫu nhiên (ví dụ kinh điển: đi men vách đá). Ưu điểm này **không thể hiện ở đây**, vì bản đồ không có ô nguy hiểm.

---

## 8. Monte Carlo ES: học sau cả episode

Hàm `_mc_chunk` gồm 3 bước:

```
1) Sinh trọn một episode:
       s ← trạng thái ngẫu nhiên;  a₀ ← ngẫu nhiên          (exploring starts)
       các bước sau: a ← argmax Q(s, ·)                      (tham lam, ε = 0)
       lưu (s_t, a_t, r_t) cho đến khi giao hết hoặc hết max_steps

2) Đánh dấu lần ĐẦU TIÊN mỗi cặp (s, a) xuất hiện trong episode   (first-visit)

3) Đi ngược từ cuối episode:
       G = 0
       với t = T−1, …, 0:
           G = r_t + γ · G                                   (return thật từ bước t)
           nếu là lần đầu của (s_t, a_t):  Q(s_t, a_t) ← Q + α · (G − Q)
```

$$G_t = r_{t+1} + \gamma \, G_{t+1}, \qquad \text{target} = G_t$$

**Ví dụ:** một episode ngắn có phần thưởng $-1, -1, +19$ (đi 2 bước rồi giao). Tính ngược:

- $G_2 = 19$
- $G_1 = -1 + 0{,}99 \cdot 19 = 17{,}81$
- $G_0 = -1 + 0{,}99 \cdot 17{,}81 = 16{,}63$

Ra cùng con số như ví dụ Q-learning, nhưng ở đây là **return thật đi hết episode**, không phải ước lượng mượn từ $Q(s')$ (không "bootstrap").

**Tham số và lý do:**

| Tham số | Giá trị | Lý do |
|---|---|---|
| $\varepsilon$ | 0 (tham lam hoàn toàn) | mọi sự khám phá đến từ exploring starts, đúng định nghĩa Monte Carlo ES |
| $\alpha$ | 1 ($Q$ = return mới nhất) | chính sách tham lam trong môi trường tất định cho return **chính xác**; lấy trung bình $1/N$ (cấu hình cũ) thì $Q$ giữ rất lâu các return rất âm từ thời chính sách còn đi lòng vòng |
| Bị cắt vì `max_steps` | không bootstrap ($G$ bắt đầu từ 0) | nếu robot lặp vòng 300 bước, return $\approx -(1 - 0{,}99^{300})/0{,}01 \approx -95$, nên hành động gây vòng lặp bị phạt nặng và lần sau không được chọn |

**Nhược điểm:** phải đợi hết episode mới học, nên giai đoạn đầu chậm hơn Q-learning (màn 4 cần 1,74 triệu episode). Monte Carlo ES **phụ thuộc hoàn toàn vào exploring starts**: thiếu nó, độ chính xác là 0% ở màn 3 và 4. Bù lại, mỗi episode tham lam thường ngắn, nên ở màn 3 nó còn nhanh hơn Q-learning một chút (trung vị 290 nghìn so với 350 nghìn episode).

---

## 9. Lịch giảm ε và α

Hàm `_schedule` trong `agents.py`:

$$\varepsilon(ep) = \varepsilon_s \cdot \left( \frac{\varepsilon_e}{\varepsilon_s} \right)^{\min\left(1,\ \frac{ep}{f \cdot n_{ep}}\right)} \qquad \text{(giảm theo hàm mũ)}$$

$\varepsilon$ đi từ $\varepsilon_s$ xuống $\varepsilon_e$ trong $f \cdot n_{ep}$ episode đầu, sau đó giữ nguyên. Giảm theo hàm mũ thì nhanh lúc đầu và chậm dần về sau.

**Ví dụ** Q-learning: $\varepsilon_s = 1$, $\varepsilon_e = 0{,}01$, $f = 0{,}8$. Ở giữa quãng giảm ($ep = 0{,}4 \cdot n_{ep}$): $\varepsilon = 1 \cdot 0{,}01^{0{,}5} = 0{,}1$.

Bước học $\alpha$ của SARSA giảm **tuyến tính** suốt quá trình huấn luyện:

$$\alpha(ep) = \alpha_s + \frac{ep}{n_{ep}} \cdot (\alpha_e - \alpha_s), \qquad 0{,}5 \rightarrow 0{,}1$$

| | Q-learning | SARSA | Monte Carlo ES |
|---|---|---|---|
| $\varepsilon$ | 1 → 0,01 (mũ, 80%) | 1 → $10^{-5}$ (mũ, 50%) | 0 |
| $\alpha$ | 1 | 0,5 → 0,1 (tuyến tính) | 1 |
| $\gamma$ | 0,99 | 0,99 | 0,99 |
| Số episode | S1: 100k · S2: 300k · S3: 2M · S4: 6M | như bên trái | như bên trái |

---

## 10. Đánh giá: độ chính xác và trạng thái sai

### Độ chính xác

$$\text{Acc} = \frac{1}{|C|} \sum_{c \in C} \mathbf{1}\big[\, R^{\pi}(c) = R^{*}(c) \,\big]$$

- $C$: tập ô xuất phát (mọi ô trống, mọi kiện đang chờ, $\tau = 0$, ô ×2 chưa dùng).
- $\mathbf{1}[\dots]$ bằng 1 nếu return của chính sách học được từ ô $c$ **đúng bằng** return tối ưu của Value Iteration, ngược lại bằng 0.

Acc là **tỉ lệ ô xuất phát mà robot đi đúng lộ trình tối ưu**. Thước đo này chặt hơn việc chỉ kiểm tra từ ô S. Ví dụ Q-learning với cấu hình cũ vẫn đi tối ưu từ S ở màn 3 trong 5/5 seed, nhưng chỉ đúng **20%** số ô xuất phát: nó "học thuộc" một lộ trình chứ chưa giải được bài toán.

**Kết quả:** cả 60 lần chạy (3 thuật toán × 4 màn × 5 seed) đều kết thúc ở 100%.

### Trạng thái sai (`failure_cases.py`)

Độ chính xác chỉ thả robot ở trạng thái "mới bắt đầu". `failure_cases.py` thả robot vào **mọi trạng thái đạt được chưa giao xong** (184 ở màn 1 đến 15.604 ở màn 4). Một trạng thái là **sai** khi chính sách thu được ít hơn return tối ưu từ đó. Mỗi trạng thái sai được xếp vào một loại: không bao giờ xong (lặp vòng), giao trễ, Pickup/Drop sai, hoặc đi đường dài hơn. Code còn lần theo lộ trình tới **quyết định sai** đầu tiên, tức trạng thái đầu tiên mà hành động tham lam không tối ưu theo $Q^*$, và ghi lại bằng mã trạng thái.

| Màn (số trạng thái) | Q-learning | SARSA | Monte Carlo ES |
|---|---|---|---|
| 1 (184) | 0,2% → 0 | 45% → 0 | 11% → 0 |
| 2 (532) | 5,9% → 0 | 31% → 0 | 37% → 0 |
| 3 (4.024) | 33% → 0 | 79% → 0,3% | 83% → 0 |
| 4 (15.604) | 36% → 0 | 83% → 1,8% | 91% → 0 |

(cấu hình cũ → cấu hình mới, trung bình 5 seed)

**Lỗi điển hình của SARSA ở màn 4:** từ trạng thái mã 643 (góc trên trái, đang mang C, ô ×2 bật, $\tau = 7$), SARSA đi 3 bước tới ô trả hàng của C (mã **17419**) và **giao C ngay**. Hành động tối ưu là đi Xuống để lấy A trong lúc vẫn giữ C. Giao C sớm kết thúc chuyến ×2, nên các kiện còn lại phải mang từng cái: **61 bước** (return $-11$) thay vì **55 bước** (return $-5$).

Lý do: trong $Q^*$, Drop và Xuống chỉ chênh **0,04** (Xuống $-9{,}89$, Drop $-9{,}93$). Với $\gamma = 0{,}99$, giao C ngay mang phần thưởng +20 về sớm, gần đủ bù 6 bước đi thêm về sau. Sai số Q của SARSA ở đó là 0,12–0,24, lớn hơn khoảng chênh, nên thứ tự hai hành động bị đảo.

- Toàn bộ 301 trạng thái sai của seed 0 đến từ chỉ 39 quyết định sai, nằm trên **hai ô**: ô lấy và ô trả của C.
- Gấp đôi số episode **không** giúp (vẫn 1,8%), nên lỗi không do huấn luyện ít.
- Giảm $\alpha$ cuối xuống 0,02 thì lỗi ở màn 4 **giảm một nửa** (0,9%) và **hết hẳn** ở màn 3. Điều này khẳng định nguyên nhân là nhiễu trong target của SARSA.

---

## 11. So sánh ba thuật toán, cấu hình cũ và cấu hình mới

| | Q-learning | SARSA | Monte Carlo ES |
|---|---|---|---|
| Cập nhật khi nào | sau mỗi bước | sau mỗi bước | sau cả episode |
| Target | $r + \gamma \max Q(s')$ | $r + \gamma Q(s', a')$ | return thật $G$ |
| Học giá trị của | chính sách tối ưu (off-policy) | chính sách ε-greedy của nó (on-policy) | chính sách tham lam của nó |
| Nhiễu trong target | không | có (do $a'$ ngẫu nhiên) | không (khi $\varepsilon = 0$) |
| Số episode tới 100% ở màn 4 (trung vị) | **1,14M** | 3,57M | 1,74M |
| Độ chính xác trung bình khi học (màn 1 → 4) | 0,99 / 0,99 / 0,95 / 0,94 | 0,77 / 0,77 / 0,60 / 0,48 | 0,95 / 0,91 / 0,89 / 0,80 |
| Tụt dưới 100% sau khi đã đạt | không bao giờ | 9–19% số lần đánh giá | không bao giờ |
| $\lvert Q - Q^* \rvert$ cuối | 0 | 0,001 – 0,047 | 0 |
| Sai khi thả ở mọi trạng thái (màn 3 / 4) | 0 / 0 | 0,3% / 1,8% | 0 / 0 |
| Thời gian 1 seed ở màn 4 | 38 s | 23 s | 38 s |

**Xếp hạng:** Q-learning nhanh và ổn định nhất; Monte Carlo ES thứ hai; SARSA chậm nhất và còn sai nhẹ ở vài trạng thái. Tính theo giây thì một episode SARSA lại rẻ nhất.

### Cấu hình cũ khác gì và vì sao thất bại

Cấu hình cũ, chạy với **cùng số episode**:
- xuất phát ở ô trống ngẫu nhiên nhưng luôn với $\tau = 0$, $b = 0$, hành động đầu không ngẫu nhiên;
- $\varepsilon$ giảm **tuyến tính** từ 1 xuống 0,05 trong 70% số episode đầu;
- $\alpha = 0{,}5$ cho Q-learning và SARSA;
- Monte Carlo lấy trung bình $1/N$.

| Màn | Q-learning | SARSA | Monte Carlo |
|---|---|---|---|
| 1 | 100% → 100% | 55,7% → 100% | 97,4% → 100% |
| 2 | 97,1% → 100% | 52,1% → 100% | 70,7% → 100% |
| 3 | 20,0% → 100% | 1,1% → 100% | 0% → 100% |
| 4 | 61,3% → 100% | 0,4% → 100% | 0% → 100% |

(độ chính xác cuối: cũ → mới)

**Nguyên nhân:** $\tau$ và $b$ nhân số trạng thái lên rất nhiều. Khi luôn xuất phát từ $\tau = 0$, phần lớn trạng thái như "đang mang C, $\tau = 15$" **gần như không bao giờ được ghé**, nên $Q$ ở đó không được học. Robot lặp vòng (đâm mãi vào tường), giao trễ, hoặc làm Pickup/Drop sai. Thêm nữa, $\varepsilon$ dừng ở 0,05 khiến SARSA học giá trị của một chính sách vẫn hành động ngẫu nhiên 5% số bước.

**Cải tiến quyết định** là exploring starts (trạng thái ngẫu nhiên + hành động đầu ngẫu nhiên), cộng với lịch $\varepsilon$ và $\alpha$ riêng cho từng thuật toán.

---

## 12. Câu hỏi thường gặp khi bảo vệ

**12.1. Vì sao không dùng DQN hay mạng nơ-ron?**
Không gian trạng thái đủ nhỏ (tối đa 15.612 trạng thái đạt được) để lưu nguyên một bảng Q, và đề bài yêu cầu phương pháp dạng bảng. Bảng Q cho kết quả chính xác, kiểm chứng được với Value Iteration, và tra được từng trạng thái bằng mã trạng thái. Mạng nơ-ron chỉ cần khi trạng thái quá nhiều hoặc liên tục.

**12.2. Vì sao phải thêm $\tau$ và $b$ vào trạng thái?**
Để giữ tính Markov (mục 2). Thiếu $\tau$ thì cùng một trạng thái có hai tương lai (kịp hoặc trễ deadline); thiếu $b$ thì kết quả Pickup phụ thuộc lịch sử.

**12.3. Vì sao Q-learning dùng $\alpha = 1$ được?**
Môi trường tất định: cùng $(s, a)$ luôn cho cùng $r$ và $s'$, nên target $r + \gamma \max Q(s')$ không có nhiễu. Gán thẳng $Q = $ target là đúng; không cần lấy trung bình.

**12.4. Vì sao SARSA chậm nhất?**
Target của SARSA dùng hành động $a'$ thật sự sẽ làm, mà $a'$ đôi khi ngẫu nhiên, nên target có nhiễu. Phải dùng $\alpha$ nhỏ dần và $\varepsilon$ gần 0 mới hội tụ về $Q^*$, mất khoảng 3 lần số episode so với Q-learning.

**12.5. Exploring starts có dùng được ngoài thực tế không?**
Khó: phải đặt được robot vào bất kỳ trạng thái nào, điều chỉ dễ trong mô phỏng. Đây là một hạn chế đã nêu trong báo cáo.

**12.6. Vì sao giá trị V tại S nhỏ hơn return tối ưu?**
Vì $\gamma = 0{,}99$ chiết khấu phần thưởng nhận muộn. Ví dụ màn 4: return tối ưu (không chiết khấu) là 13, nhưng $V(S) = 2{,}5$ vì các phần thưởng $+20$ đến ở bước 22, 33, 65, 67.

**12.7. Episode bị cắt vì `max_steps` xử lý thế nào?**
Bị cắt là giới hạn thời gian, không phải trạng thái kết thúc thật. Q-learning và SARSA vẫn bootstrap ở bước cuối ($r + \gamma Q(s')$) chứ không coi giá trị tương lai bằng 0. Monte Carlo thì cố ý không bootstrap, để vòng lặp nhận return rất âm và bị loại (mục 8).

**12.8. Độ chính xác khác tỉ lệ thành công thế nào?**
Thành công chỉ cần giao hết hàng trước `max_steps`. Độ chính xác đòi hỏi return **đúng bằng** tối ưu, tức đường ngắn nhất và không trễ deadline, từ **mọi** ô xuất phát. Một chính sách có thể thành công 100% mà độ chính xác vẫn thấp.

**12.9. Thay đổi phần thưởng thì sao?**
Ở màn 3, bỏ phạt trễ (30 → 0) hoặc nới deadline (20 → 45) thì robot đổi sang thứ tự B → A → C (40 bước). Bỏ ô ×2 ở màn 4 thì robot đi 4 chuyến riêng (D → B → C → A, 75 bước). Mọi trường hợp đều khớp đáp án tối ưu ở 5/5 seed.

**12.10. Hướng phát triển?**
Môi trường ngẫu nhiên (bánh xe trượt) để xem thứ hạng thuật toán có đổi không; thử Expected SARSA và Q(λ); dùng xấp xỉ hàm để một mô hình dùng được cho nhiều bản đồ.

---

## 13. Bảng ký hiệu: toán ↔ code

| Ký hiệu | Ý nghĩa | Trong code |
|---|---|---|
| $s$, $s'$ | trạng thái hiện tại, trạng thái kế | `s`, `s2` (số nguyên = mã trạng thái) |
| $a$, $a'$ | hành động hiện tại, hành động kế | `a`, `a2` (0–5: Lên, Xuống, Trái, Phải, Pickup, Drop) |
| $r$ | phần thưởng một bước | `R[s, a]` |
| done | bước đó giao hết hàng | `D[s, a]` |
| $Q(s,a)$ | bảng giá trị hành động | `Q[s, a]` |
| $Q^*$, $V$ | đáp án tối ưu từ Value Iteration | `value_iteration()` trong `planner.py` |
| $\gamma$ | hệ số chiết khấu (0,99) | `gamma` |
| $\alpha$ | bước học | `alpha`, `alpha_end` |
| $\varepsilon$ | xác suất khám phá | `eps_start`, `eps_end`, `decay_frac`, `eps_decay` |
| $\sigma_i$ | trạng thái kiện $i$ (0 chờ, 1 mang, 2 đã giao) | `status[i]` |
| $\tau$, $T_d$ | đồng hồ deadline, hạn deadline | `tau`, `deadline` |
| $b$ | trạng thái ô ×2 | `boost` |
| idx | mã trạng thái | `encode()` / `decode()` trong `delivery_env.py` |
| $G_t$ | return từ bước $t$ (Monte Carlo) | `G` trong `_mc_chunk` |
| Acc | độ chính xác | `eval_acc` trong lịch sử huấn luyện |

Xem thêm: `README.md` (giải thích từng file code, tiếng Anh), báo cáo Word và bài báo IEEE trong `Nguyen_Quang_Trung-22011211/`.
