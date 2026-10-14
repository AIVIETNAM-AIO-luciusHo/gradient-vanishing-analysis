# TASK_REPORT — ĐỢT 2 (Coder → Lead)

Báo cáo Đợt 1 đã lưu ở `TASK_REPORT_DOT1.md` (copy trước khi ghi). Không commit/add/push. Không sửa file cấm
(`git diff --stat` trên results_*, experiment cũ, figures: rỗng). Build cuối: `bash build.sh` rc=0.
Interpreter cho thí nghiệm mới: `../../../code-test/.venv/bin/python` (torch 2.14.0+cpu, Python 3.12.3, khớp header results.md). `/usr/bin/python3` không có torch.

## 1. File đã sửa / tạo
| File | Tóm tắt |
|---|---|
| `experiment_jacobian.py` (MỚI) | Phổ J=∂z^[L-1]/∂z^[1] (64×64, bỏ tầng ra); import `build/seed_all/CONDS/D/N/SEEDS/DEPTHS` từ `experiment`; init float32 → `.double()` chỉ cho phổ; 16 mẫu; assert B4(a)(b)(c)(d). |
| `results_jacobian.md` (MỚI) | Bảng 6 hàng (A/B × L=5,10,20) + header; `cmp` hai lần chạy giống từng byte. |
| `sections/03-nguyennhan.tex` | Thêm `\subsection{Dynamical Isometry…}` (`subsec:dyniso`): định nghĩa J, Mệnh đề + chứng minh 3 bước, hệ quả, Dynamical Isometry, khởi tạo trực giao, Yang; sửa đoạn "ba lối thoát…vét cạn" → "ba lối thoát chính" + nhắc `subsec:dyniso` (cũng sửa câu liền trước "chỉ ra chính xác ba lối thoát còn lại" → "ba lối thoát chính"). |
| `sections/06-thucnghiem.tex` | Thêm `\subsection{Phổ Jacobian đầu-cuối…}` + Bảng `tab:jacobian-spectrum`; thêm mục (d) (`par:khoitao-bn`); D-a (bỏ in đậm epoch 0, caption "từ epoch 1"); D-b (hai Overfull). |
| `sections/07-tongket.tex` | Viết lại đoạn BatchNorm/LayerNorm (đủ 5 ý (i)–(v)); sửa câu mở đầu mục cho nhất quán C1(6). |
| `references.bib`, `build.sh`, `main.tex` | Không đổi trong Đợt 2. Cả Yang2019MeanField và Xiao2018DynIso đã được `\cite`. |

## 2. Nghiệm thu E1–E8
| # | KQ | Lệnh / output rút gọn |
|---|---|---|
| E1 | **PASS** | `grep -n 'cite.*Yang2019MeanField' sections/*.tex` → 03:500, 06:545/560/565, 07:100; `…Xiao2018DynIso` → 03:473/492, 07:107. `pdftotext main.pdf -`: `[8] L. Xiao et al., "Dynamical isometry and a mean field theory of CNNs…", ICML, 2018, pp. 5393–5402.` và `[9] G. Yang et al., "A mean field theory of batch normalization", ICLR, 2019.` `grep -a -c "Citation \`" main.log`=0; `grep -c '\[?\]\|??'`=0. |
| E2 | **PASS** | Script D1: 12 nhãn đều PASS (gồm `tab:jacobian-spectrum`, `fig:gradnorm`). `grep -nE '(Bảng\|Hình)[ ~]*[0-9]'` rỗng. Bảng mới: caption box (`\captab`) đứng trước `\begin{tabular}`. |
| E3 | **PASS** | Xem mục 3. `python3 experiment_jacobian.py` hai lần → `cmp results_jacobian.md` IDENTICAL, `cmp` hai log IDENTICAL; in `B4a OK / B4b OK / B4c OK / B4d OK`. |
| E4 | **PASS (chờ Lead đọc)** | Mục 4. |
| E5 | **PASS** (D-c: xem mục 6) | `bash build.sh` rc=0; `grep -a -c Overfull main.log`=0; `grep -c '^!' main.log`=0; không "Reference/Citation … undefined". Cảnh báo còn lại: biblatex fall-back/csquotes/Language vietnamese, pgfplots compat, Font Warning 5.5/16, "undefined references" + "(re)run BibTeX" (D-c). |
| E6 | **PASS** | `git diff --stat -- results_training.json results_training.md results.md experiment.py experiment_training_dynamics.py experiment_numpy_legacy.py figures` rỗng. `git status`: M main.pdf, main.tex, references.bib (từ Đợt 1), sections/02,03,04,06,07; ?? experiment_jacobian.py, results_jacobian.md, TASK_*.md, sections/08-phuluc.tex, và 3 file untracked có sẵn (FEEDBACK, experiment_stdout.txt, training_dynamics_sanity.log). |
| E7 | **PASS** | Trang: **33 → 38**. Thân bài vẫn 4 hình (Hình 1–4), Hình 5 ở Phụ lục A. Bảng 1–7 liền mạch (Bảng Jacobian = Bảng 5; training-dynamics → 6; nguyên-lý → 7). Đã xem ảnh trang 12, 13 (mệnh đề, chứng minh), 27 (bảng mới): không tràn lề. +5 trang do: mục III.5 (~2.5 tr.), mục Jacobian + bảng (~1 tr.), mục (d), đoạn BN ở 07, và dịch phao. |
| E8 | **PASS** | `grep -n -i 'adam\|R\^2\|decay_slope\|Hạn chế' sections/*.tex` → rỗng. |

## 3. Đối chiếu số liệu (E3) và assert B4
- **Bảng `tab:jacobian-spectrum` ↔ `results_jacobian.md`**: script đối chiếu từng ô (5 cột × 6 hàng = 30 ô + nhãn n/a) so với cột tương ứng của dòng cùng (điều kiện, L): **tất cả khớp** (log10 σ_max ↔ col 3; log10 RMS ↔ col 4; hạng ↔ col 5; log10 κ ↔ col 7 hoặc n/a; log10 c_J ↔ col 8). Ghi chú n/a "0/80" ↔ col 6 "0/80".
- **Số trong mục (d) ↔ `results_training.md`** (từng chuỗi `mean ± std` được tìm thấy trong file): `0.862 ± 0.0482`, `1.56 ± 0.0611`, `2.81 ± 0.0832` (log10ρ epoch 0, plain relu+kaiming BN on, bảng "log10(rho)"); `8.22 ± 0.32`, `36.8 ± 1.07`, `626 ± 59.4` (bảng "stem ||g||_F"); `0.0232 ± 0.00475`, `-0.297 ± 0.0768`, `-1.3 ± 0.0419` (epoch 1); `0.332 ± 0.0716`, `-0.0106 ± 0.144`, `-1.26 ± 0.145` (epoch 10); `-0.0372 ± 0.0977`, `0.0626 ± 0.139`, `-0.133 ± 0.191` (plain không BN, epoch 0); `-0.16 ± 0.0756`, `-0.0732 ± 0.119`, `-0.0524 ± 0.0885` (plain sigmoid+xavier+BN); `2.95 ± 0.9` (stem gnorm L=32 không BN; cũng là ô trong Bảng training-dynamics). Khoảng ResNet+BN `[-0.914, 0.348]` = min/max của 12 giá trị (3 L × epoch 0,1,5,10) trong bảng; tôi đã diễn đạt "ở cả ba độ sâu tại các epoch 0,1,5,10" (câu (c) cũ nói "11 lần đo" ở L=32, khác tập đo nhưng cùng khoảng).
- **Số suy ra trong văn bản (không ghi trong file nguồn)**: $1/(64\epsilon_{64})\approx7\times10^{13}$ (từ định nghĩa ngưỡng, ε64=2.2e-16); "hạng ≈ 24–29/64" (làm tròn 23.6–28.9); "$10^{5.7}$, $10^{11.6}$" (từ 5.72, 11.6). Không dùng số suy ra khác.
- **Assert B4** (đều "OK" trong log): (a) σ_max(J) ≤ c_J(1+1e-9) cho cả 5×3×2×16 mẫu; (b) `autograd` ∂(vᵀz^[L-1])/∂z^[1] = Jᵀv, đồng thời tích tường minh D¹W²ᵀD²…D^{L-2}W^{L-1}ᵀv = Jᵀv (rtol 1e-9; sigmoid L=10: sai lệch max 2.7e-20, ReLU L=10: 2.7e-15); (c) chạy lặp (42,5,·) cho kết quả giống hệt; (d) `c_ave` (float32, như experiment.py) khớp chuỗi 3 chữ số của cột "c (bound)" ở cả 6 hàng (vd. A L=5 `0.449 ± 0.00761`, B L=20 `2.68 ± 0.0141`).
- **Quan sát đáng chú ý**: ReLU: hạng số học 23.6–28.9/64 và 0/80 cặp đủ hạng ở mọi L; Sigmoid L=20: hạng đúng 61 ở mọi hạt giống và mẫu (ngưỡng 64·ε64·σ_max cắt đồng nhất vì phổ giảm theo cấp số nhân), 0/80 đủ hạng → κ = n/a; L=5,10: 80/80.

## 4. Mệnh đề C1(2) và chứng minh (E4)
Giả thiết: các tầng ẩn 1…L−1 cùng rộng n, L ≥ 3, `D^[k]=diag(σ'(z^[k]))`.
- Định nghĩa (eq:Jdef): `J := ∂z^[L-1]/∂z^[1] = W^[L-1] D^[L-2] ⋯ W^[2] D^[1] ∈ R^{n×n}` (L−2 thừa số `W^[k]D^[k-1]`, đúng eq:jacforward theo quy tắc dây chuyền).
- **Mệnh đề**: với mọi δ^[L-1] ∈ R^n,
  (eq:deltaJ) `δ^[1] = D^[1](W^[2])ᵀ D^[2](W^[3])ᵀ ⋯ D^[L-2](W^[L-1])ᵀ δ^[L-1] = Jᵀ δ^[L-1]`,
  (eq:twoside) `σ_min(J)‖δ^[L-1]‖ ≤ ‖δ^[1]‖ ≤ σ_max(J)‖δ^[L-1]‖`, hai dấu bằng đều đạt được.
- **Chứng minh**: B1 đặt `P_l := D^[l](W^[l+1])ᵀ D^[l+1] ⋯ D^[L-2](W^[L-1])ᵀ`; quy nạp lùi theo l: l=L−2 là eq:deltarec, bước l→l−1 là `δ^[l-1]=D^[l-1](W^[l])ᵀδ^[l]` (eq:deltarec) ⇒ `δ^[l]=P_lδ^[L-1]`, tới l=1. B2: D đối xứng và chuyển vị đảo thứ tự ⇒ `Jᵀ=P_1`. B3: `J=UΣVᵀ` ⇒ `‖Jᵀδ‖²=Σσ_i²(u_iᵀδ)²` với `Σ(u_iᵀδ)²=‖δ‖²` ⇒ kẹp giữa σ_min²‖δ‖² và σ_max²‖δ‖²; dấu bằng tại δ=u_n, u_1.
- **Hệ quả** (nêu trong văn bản): `σ_max(J) ≤ ∏_{k=2}^{L-1} ‖W^[k]‖₂ max|σ'(z^[k-1])|` (chặn kiểu c^{L-2}, eq:governing); c chỉ cho vế phải, còn σ_min/κ(J) mới cho biết mọi hướng có về tới hay không. Ghi chú: g^[1]=δ^[1]a^[0]ᵀ nên so với chuẩn gradient chỉ ở bậc độ lớn.
- **Khớp thứ tự/chuyển vị**: eq:jacforward (W^[k] D^[k-1], D bên phải) ↔ script `forward_hidden` (z=W_k(act(z))) ↔ `check_vjp` (tích tường minh `d = dact(z_{k-1}) * (W_kᵀ d)` từ k=L−1 xuống 2) — B4b xác nhận bằng autograd.

## 5. Kiểm chứng phát biểu về Yang2019 và Xiao2018
Nguồn: tóm tắt arXiv lấy bằng API `export.arxiv.org/api/query?id_list=…` (nguyên văn).
**Yang (1902.08129)**: "We develop a mean field theory for batch normalization in fully-connected feedforward neural networks. … Our theory shows that gradient signals grow exponentially in depth and that these exploding gradients cannot be eliminated by tuning the initial weight variances or by adjusting the nonlinear activation function. Indeed, batch normalization itself is the cause of gradient explosion. As a result, vanilla batch-normalized networks without skip connections are not trainable at large depths for common initialization schemes…"
- Dùng: mạng truyền thẳng kết nối đầy đủ, rộng, tại khởi tạo ("wide … at initialization" cũng trong tóm tắt); gradient tăng theo hàm mũ độ sâu; không loại bỏ được bằng chỉnh phương sai khởi tạo hay đổi hàm kích hoạt; BN là nguyên nhân; mạng BN thuần không kết nối tắt không huấn luyện được ở độ sâu lớn với khởi tạo thông dụng. Cụm "bất kể hàm kích hoạt" tôi diễn đạt đúng theo "cannot be eliminated by … adjusting the nonlinear activation function".
- **BỎ vì không kiểm chứng được**: vế "kết nối tắt làm giảm hiện tượng này" của C1(5): tóm tắt chỉ nói BN thuần *không có* skip connection khó huấn luyện, và gradient explosion giảm khi "tuning the network close to the linear regime"; không nói skip connection làm giảm.

**Xiao (1806.05393)**: "…we demonstrate that it is possible to train vanilla CNNs with ten thousand layers or more simply by using an appropriate initialization scheme. … by characterizing the conditions for dynamical isometry, the equilibration of singular values of the input-output Jacobian matrix. These conditions require that the convolution operator be an orthogonal transformation in the sense that it is norm-preserving. We present an algorithm for generating such random initial orthogonal convolution kernels…"
- Dùng: CNN thuần (không residual/BN: "While residual connections and batch normalization do enable training at these depths, it has remained unclear whether such specialized architecture designs are truly necessary"), ≥10 000 tầng chỉ nhờ khởi tạo; dynamical isometry = "equilibration of singular values of the input-output Jacobian"; điều kiện buộc toán tử tích chập trực giao/bảo toàn chuẩn.
- **Không có trong tóm tắt, nên ghi rõ trong văn bản là cách diễn đạt của tài liệu**: định nghĩa "mọi σ_i ≈ 1". Không phát biểu gì về ReLU/CNN ngoài tóm tắt.

## 6. D-a, D-b, D-c
- **D-a (PASS)**: bỏ `\mathbf` ở hai ô epoch 0 (PlainNet 10.6 khối BN; 9.94 khối không BN). Còn in đậm: ResNet ở epoch 1, 5, 10 của khối BN (18.3, 25.8, 32.0). Khối không BN: ResNet phân kỳ ⇒ không in đậm. Caption: "In đậm: độ chính xác test trung bình cao hơn, từ epoch 1, chỉ khi cả hai cùng không phân kỳ."
- **D-b (PASS, 0 Overfull)**: (i) 26.47pt ở "Phép đo": nguyên nhân thực sự là dòng "…forward/backward riêng" không có chỗ ngắt (dấu `/`), không phải công thức inline; sửa bằng `forward/\allowbreak backward` và (kèm) đưa 4 định nghĩa g_stem, g_head, stem gnorm, ρ vào một công thức hiển thị (nghĩa/số không đổi). (ii) 9.45pt ở bullet "ρ của ResNet ở gần 1": bọc itemize trong `{\emergencystretch=3em …}`. Overfull mới do Đợt 2 (3 chỗ: chứng minh 9.9pt, đoạn Dynamical Isometry 3.0pt, + 2 cũ) xử lý bằng: đưa định nghĩa P_l ra công thức hiển thị và bọc mục `subsec:dyniso` trong `{\emergencystretch=3em …}`. Cuối cùng `grep -a -c Overfull main.log`=0.
- **D-c ("There were undefined references")**: KHÔNG phải reference/citation thật. Bằng chứng: (1) không có "Reference/Citation … undefined" nào trong main.log; PDF không có `??`/`[?]`; main.aux có đủ `\abx@aux@cite` cho 9 khoá và main.bbl có 9 `\entry`; refcontext aux (`none/global//global/global`) = `\datalist` trong bbl. (2) Có sẵn ở bản gốc (build `git archive HEAD`, Đợt 1). (3) Chạy thêm 2 vòng `bibtex`+`pdflatex` không làm hết (chạy lại không có tác dụng). (4) Thử nghiệm trong bản sao ngoài repo: xoá TOÀN BỘ `\cite` ⇒ cả 2 cảnh báo biến mất; giữ lại chỉ cite của 05-bptt (2 cite) hoặc chỉ 01-tongquan vẫn còn cảnh báo ⇒ nó xuất hiện khi có bất kỳ `\cite` nào với `backend=bibtex` (log báo "Using fall-back bibtex backend"), không do một khoá/tệp cụ thể; đổi `citestyle` sang `numeric`, `backend=bibtex8`, bỏ `\printbibliography` đều không hết. (5) Riêng "Please (re)run BibTeX on main" do các tuỳ chọn `maxnames=1, minnames=1, uniquelist=false`: bỏ chúng thì thông báo này biến mất (vẫn còn dòng "undefined references"). Căn nguyên cơ chế bên trong biblatex của dòng "undefined references" tôi chưa xác định được; kết luận chỉ là "vô hại, có bằng chứng". Không sửa `build.sh`/preamble.

## 7. Điểm tự quyết khác prompt / cần Lead–User quyết
1. **C4(v) lệch nhẹ so với prompt**: prompt viết "chỉ tổ hợp ResNet+BN giữ ρ gần 1". Nhưng notebox cuối 06 đã nêu Sigmoid + BN (cả Plain lẫn ResNet) cũng có log10ρ ∈ [−0.41, 0.09]. Để không tự mâu thuẫn, tôi giới hạn câu trong "nhóm ReLU + Kaiming ở L=32 của Bảng training-dynamics": chỉ ResNet + BN vừa giữ ρ gần 1 vừa để stem học được, với độ chính xác cao nhất trong bảng. Cần Lead xác nhận.
2. **Trích Xiao**: định nghĩa "mọi σ_i ≈ 1" là của tài liệu (đã ghi rõ trong văn bản), không trích từ tóm tắt.
3. **C1(3)**: theo prompt dùng "(G1)–(G3)" cho nhận xét RMS(σ)↔bảo toàn phương sai; số đo ủng hộ ở ReLU + He (RMS≈1) nhưng Xavier + sigmoid có RMS≪1, nên văn bản nói Xavier không giữ được đại lượng trung bình này qua sigmoid (kèm "hệ số σ'(0)=1/4 và độ bão hoà không được tính vào s²", là giải thích, không đo riêng). Mục (d) dùng "(G1)–(G4)" theo C3.
4. Thêm một câu ở 03 ("chỉ ra chính xác ba lối thoát còn lại" → "ba lối thoát chính"), ngoài đoạn C1(6) nhưng cùng khẳng định "vét cạn".
5. Dùng `{\emergencystretch=3em …}` cục bộ (không đổi preamble) để triệt Overfull.
6. Số "$1/(64\epsilon_{64})\approx7\times10^{13}$" là hệ quả toán của định nghĩa ngưỡng, không phải số đo.
7. Số trang 38 (+5): có thể rút bớt nếu Lead muốn (ví dụ gộp bớt đoạn phân tích Bảng 5).
8. Dừng sau báo cáo; chờ Lead nghiệm thu trước Đợt 3.
