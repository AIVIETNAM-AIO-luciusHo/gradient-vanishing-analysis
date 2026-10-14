# TASK_REPORT — ĐỢT 3 (Coder → Lead)

Báo cáo Đợt 2 đã lưu ở `TASK_REPORT_DOT2.md` (copy trước khi ghi). Không commit/add/push (`git log` vẫn 3 commit cũ). Không chạy lại thí nghiệm nào; chỉ ĐỌC `results_training.json`.
Interpreter: `../../../code-test/.venv/bin/python` (numpy + json).

## 1. File đã sửa / tạo
| File | Tóm tắt |
|---|---|
| `analysis_decay.py` (MỚI) | Đọc `results_training.json`; TD, decay_slope, R² theo (arch, L, act, bn, epoch∈{0,1,5,10}) mean ± std (ddof=1), `(n=k)`; loại gnorm ≤0/không hữu hạn (không có phần tử nào bị loại); assert TD tính lại = `log10_rho` trong JSON và khớp từng ô với bảng log10(rho) của `results_training.md`. |
| `results_decay.md` (MỚI) | Bảng 96 hàng (24 cấu hình × 4 epoch) + header định nghĩa. |
| `sections/06-thucnghiem.tex` | Định nghĩa TD/decay_slope/R² (một chỗ, "Phép đo"); `tab:total-degradation` (Bảng 7) + đoạn "Đọc decay_slope cùng R²"; `\subsection{Hạn chế của nghiên cứu}` (`subsec:han-che`, 6 ý + đoạn Adam); làm dịu khẳng định cơ chế; heatmap `[H]` + `width=0.82\linewidth`. |
| `sections/07-tongket.tex` | 2 chỗ làm dịu (mục 4). |
| `sections/03-nguyennhan.tex` | 1 chỗ làm dịu (mục 4). |
| `main.tex`, `references.bib`, `sections/08-phuluc.tex` | Không đổi trong Đợt 3 (không thêm bib; không cite Kingma & Ba). |

## 2. Nghiệm thu F1–F8
| # | KQ | Lệnh / output rút gọn |
|---|---|---|
| F1 | **PASS** | `bash build.sh` rc=0; `grep -a -c Overfull main.log`=0; `grep -c '^!' main.log`=0; `grep -a -c 'Reference .* undefined\|Citation .* undefined' main.log`=0; `pdftotext main.pdf - \| grep -c '??\|\[?\]'`=0. (Dòng "There were undefined references" vẫn như cũ: vô hại, căn nguyên chưa xác minh, không làm thêm theo E3.) |
| F2 | **PASS** | Script D1 trên `sections/0[1-8]-*.tex`: 13 nhãn đều PASS (gồm `tab:total-degradation`, `tab:jacobian-spectrum`, `fig:gradnorm`), 0 FAIL. `grep -nE '(Bảng\|Hình)[ ~]*[0-9]' sections/*.tex main.tex` rỗng. Bảng mới: caption box (`\captab`) trước `\begin{tabular}`. Lead-in `Bảng~\ref{tab:total-degradation}` ở đoạn "Kết quả được trình bày…" (trước bảng). |
| F3 | **PASS** | Trang chứa phao (số từ/trang, số phao): p6, p7, p8, p17, p25 (2), p27, p30 (2), p31, p34, p36, p39 — mọi trang ≥340 từ, **không phao nào đứng một mình**. Trang mới/di chuyển: p29 (lead-in), **p30 = Bảng 6 + Hình 3** (trước đó Hình 3 đứng một mình trên p30; đã đổi heatmap thành `[H]`, `width=0.82\linewidth` để Bảng 6 và Hình 3 cùng trang; p29 chừa ~20% trống cuối trang), p31 = Hình 4 + text, **p34 = Bảng 7** (`tab:total-degradation`) + text, p36 = Bảng 8. Đã xem ảnh p29, 30, 31, 36. Thử `[H]` đầy đủ `width=\linewidth` gây "Overfull \vbox 52pt" nên bỏ. |
| F4 | **PASS** | Script đối chiếu: Bảng 7 trong `.tex` ↔ `results_decay.md`: **48/48 ô khớp** (8 hàng × 6 cột TD/slope/R²). `analysis_decay.py` in `assert OK … every (run, epoch)` và `assert OK: TD cells match results_training.md: 96/96`. Chạy 2 lần: `cmp results_decay.md` giống từng byte (và `cmp` hai log giống). |
| F5 | **PASS** | `git diff --stat -- results.md results_training.json results_training.md results_jacobian.md experiment.py experiment_training_dynamics.py experiment_jacobian.py experiment_numpy_legacy.py figures build.sh` → rỗng. Không commit. |
| F6 | **PASS** | `grep -n 'Hạn chế' sections/*.tex` → đúng 1 dòng (06:654 `\subsection{Hạn chế của nghiên cứu}`). 6 ý đủ; ý (1) có câu "Khảo sát quét tốc độ học (η∈{0.01,0.03,0.1}) là hướng mở ngoài phạm vi… không được thực hiện và không có số liệu". Ý (4) sửa theo repo (xem mục 7). |
| F7 | **PASS** | `grep -n -i 'adam' sections/*.tex`: chỉ trong mục Hạn chế (06:671 ý (5), 676–690 đoạn Adam). `grep -n '0\.858\|1\.12\|0\.47'`: chỉ 1 trùng hợp **không phải Adam** — 06:107 `0.479` (ô c của Bảng 3, số cũ). Công thức `\Delta\theta_i=-\eta\,\hat m_i/(\sqrt{\hat v_i}+\epsilon_{\mathrm A})`, `\epsilon_{\mathrm A}=10^{-8}`; có "từng phần tử" và tiêu đề "thảo luận lý thuyết, không phải quan sát"; "Frobenius" chỉ xuất hiện trong câu nói KHÔNG dùng nó (06:684). |
| F8 | **PASS** | `grep -n 'chính là\|là lý do\|nguồn của\|chứng tỏ' sections/06*.tex` → còn 1 kết quả (06:186 "đó là lý do cần báo cáo trung bình ± độ lệch chuẩn"): **giữ nguyên** vì là lý do phương pháp luận cho cách báo cáo, không phải khẳng định nhân quả về số liệu. Bảng "cũ → mới" ở mục 4. Yang/Xiao: không đổi so với Đợt 2 (các câu `\cite` ở 03:473/492/500, 06, 07 giữ nguyên). |

## 3. Đối chiếu số (F4) + trích bảng chính `results_decay.md` (L=32)
Đối chiếu: mỗi ô Bảng 7 ↔ dòng `(arch, 32, act, bn, epoch)` của `results_decay.md`, cột TD/decay_slope/R²; cột TD ↔ bảng "log10(rho)" của `results_training.md` (96/96 ô, assert trong script). Số bị loại khỏi hồi quy: **0** (0/14168 phần tử gnorm ≤0 hoặc không hữu hạn). Trích:

| biến thể | epoch | TD Plain | TD ResNet | slope Plain | slope ResNet | R² Plain | R² ResNet |
|---|---|---|---|---|---|---|---|
| ReLU+Kaiming+BN | 0 | 2.81 ± 0.0832 | 0.348 ± 0.101 | −0.0792 ± 0.00161 | −0.0239 ± 0.000603 | 0.978 ± 0.00236 | 0.402 ± 0.0194 |
| ReLU+Kaiming+BN | 10 | −1.26 ± 0.145 | −0.342 ± 0.156 | 0.00797 ± 0.00121 | −0.00728 ± 0.00231 | 0.196 ± 0.0597 | 0.127 ± 0.0757 |
| ReLU+Kaiming (không BN) | 0 | −0.133 ± 0.191 | −0.737 ± 0.0462 | 0.0111 ± 0.00588 | 0.0293 ± 0.00214 | 0.576 ± 0.203 | 0.934 ± 0.015 |
| ReLU+Kaiming (không BN) | 10 | 0.0497 ± 0.545 (n=2) | n/a (3/3 phân kỳ) | 0.000233 ± 0.00769 (n=2) | n/a | 0.38 ± 0.199 (n=2) | n/a |
| Sigmoid+Xavier+BN | 0 | −0.0524 ± 0.0885 | −0.242 ± 0.078 | −0.00678 ± 0.000826 | −0.00732 ± 0.00112 | 0.177 ± 0.0408 | 0.131 ± 0.0178 |
| Sigmoid+Xavier+BN | 10 | 0.0398 ± 0.157 | −0.0447 ± 0.131 | −0.00385 ± 0.00171 | −0.0073 ± 0.00171 | 0.186 ± 0.163 | 0.324 ± 0.173 |
| Sigmoid+Xavier (không BN) | 0 | −20.5 ± 0.101 | −10.8 ± 0.103 | 0.627 ± 0.00313 | 0.325 ± 0.000483 | 1 ± 3.31e-05 | 0.998 ± 0.000174 |
| Sigmoid+Xavier (không BN) | 10 | −20.1 ± 0.132 | −10.3 ± 0.0984 | 0.624 ± 0.00149 | 0.325 ± 0.00241 | 1 ± 8.52e-05 | 0.997 ± 0.000608 |

Nhận xét A4 trong văn bản (chỉ mô tả): R²≈1 ở Sigmoid+Xavier không BN, slope ResNet ≈ một nửa PlainNet; R² thấp ở các ô có BN, ngoại lệ PlainNet ReLU+BN epoch 0 (R²=0.978, slope âm). Ví dụ "gnorm không đơn điệu" lấy từ JSON (PlainNet ReLU+BN, hạt giống 42, epoch 10: các tầng giữa gần phẳng, stem thấp hơn hẳn tầng kế cận, head cao hơn) — đã kiểm tra vector gnorm đó, văn bản không đưa số mới.

## 4. Bảng "dòng cũ → dòng mới" (D1) và các chỗ giữ nguyên
| # | File | Cũ → Mới |
|---|---|---|
| 1 | 06 (Phân tích) | "khoảng cách đó **chính là** hai yếu tố (a) bão hoà và (b) lệch hướng kỳ dị mà c bỏ qua" → "đó **phù hợp với** hai yếu tố … (thí nghiệm này không tách riêng đóng góp của từng yếu tố)" |
| 2 | 06 (Jacobian) | "**Đây là lý do** một mạng He có RMS(σ)≈1 vẫn không đạt dynamical isometry" → "Một ma trận thiếu hạng không thể có mọi σ_i≈1, nên mạng He ở đây không đạt dynamical isometry dù RMS(σ)≈1" (lập luận toán + hạng đo được) |
| 3 | 06 (Jacobian) | "hạng giảm dần khi L tăng (mỗi D thêm một ràng buộc hạng)" → "…, **phù hợp với** việc mỗi D thêm một ràng buộc hạng" |
| 4 | 06 caption Hình 4 | "nên huấn luyện không tự sửa được vanishing gradient" → "tức trong thiết lập này huấn luyện không làm thay đổi đáng kể vanishing gradient" |
| 5 | 06 (a) Kết luận | "chia đôi số mũ … muốn identity path thực sự là đường cao tốc, activation bão hoà **phải** nằm trong nhánh F" → "**xấp xỉ** chia đôi … **gợi ý** … **nên** nằm trong nhánh F; thiết kế đó không được thí nghiệm ở đây" |
| 6 | 06 (b) Bằng chứng | "**Kết quả là** 9/9 lần chạy phân kỳ" → "Cả 9/9 lần chạy phân kỳ" |
| 7 | 06 (b) Kết luận | "phép cộng đó **tự nó** cộng dồn thành bùng nổ" → "Số liệu **phù hợp với cách đọc rằng** … Phương sai forward không được đo, nên đây là diễn giải" |
| 8 | 06 tiêu đề (c) | "giữ ρ gần 1 và **chuyển thành** độ chính xác" → "**đi cùng** ρ gần 1 và độ chính xác cao hơn" |
| 9 | 06 (c) bullet | "stem của ResNet **học nhanh hơn** khoảng 120 lần" → "cập nhật tương đối của stem ResNet lớn hơn khoảng 120 lần" |
| 10 | 06 (c) Kết luận | "cần **cả hai** thì stem mới học được ở L=32, và đó là **nguồn của** khoảng cách 9 điểm phần trăm" → "Số liệu phù hợp với cách đọc … Trong lưới 2×2 ở L=32, chỉ tổ hợp có cả hai cho độ chính xác test cao nhất (ResNet không BN phân kỳ, PlainNet không BN thấp hơn), và tổ hợp đó **đi cùng** khoảng cách 9 điểm phần trăm so với PlainNet+BN. Đây là mô tả, không phải phân tích trung gian hay kiểm định thống kê; 3 hạt giống trong một cấu hình không độc lập" |
| 11 | 06 (a),(b),(c) | "*Cơ chế.*" → "*Cơ chế* (diễn giải lý thuyết)." (thêm nhãn, phù hợp ý (4) của Hạn chế) |
| 12 | 03 (tanh) | "**Đây là lý do** việc chuyển từ sigmoid sang tanh giúp … thất bại ở 50 tầng" → "Điều này **gợi ý vì sao** … **có thể** giúp …; tài liệu không thí nghiệm với tanh" |
| 13 | 07 (kết nối tắt) | "Đây là một lý do quan trọng **khiến** He và cộng sự huấn luyện được mạng 152 tầng" → "Điều này **phù hợp với việc** He và cộng sự huấn luyện được mạng 152 tầng" |
| 14 | 07 (guideline) | "đây là hai thay đổi rẻ nhất, **thường đủ**" → "… rẻ nhất về chi phí tính toán, và **có thể** đủ" |

**Đã rà, giữ nguyên:** 06 "đó là lý do cần báo cáo trung bình ± độ lệch chuẩn" (lý do phương pháp luận); 06 guidelinebox "Mọi tầng cùng bậc độ lớn ⇒ … nên tìm nguyên nhân ở chỗ khác" (lời khuyên gỡ lỗi, không phải phát biểu về số liệu); 06 Phép đo/Giao thức "nên A và B thấy cùng một lô dữ liệu", "Vì vậy ta đo…" (thiết kế thí nghiệm/đại số); 06 (Jacobian) "rank(J) ≤ rank(D^[1])" (toán); 06 notebox "BatchNorm một mình đã xoá vanishing gradient của sigmoid về mặt chuẩn gradient" (so sánh trực tiếp hai cấu hình chỉ khác BN trong cùng lưới, đo log10ρ); 06 (d) Cơ chế "huấn luyện làm phân phối trọng số trôi đi" (đã gắn nhãn diễn giải); 03 "Đây chính là lý do khởi tạo trọng số là một quyết định thiết kế" (kết luận thiết kế từ chặn đã chứng minh); 03 "khiến câu hỏi về c trở nên không còn quyết định" ("quyết định" = quyết định được, toán); 03:493 "chỉ nhờ một sơ đồ khởi tạo" (trích tóm tắt Xiao); 07 "một trong những can thiệp rẻ nhất: không tốn thêm phép tính sau bước khởi tạo" (chi phí tính toán); 07 "không kết luận … rẻ nhất hay bền nhất" (đã là phát biểu thận trọng).

## 5. Quyết định về Spearman (D2): **BỎ**
Đã thử tính (TD epoch 0 ↔ test acc epoch 10, run không phân kỳ, hạng trung bình khi hoà): gộp n=62 cho ρ_s=0.724, nhưng trong từng nhóm (act, bn): ReLU không BN n=8 → 0.167; ReLU+BN n=18 → −0.387; Sigmoid không BN n=18 → không xác định (mọi test acc đều 10%, phương sai 0); Sigmoid+BN n=18 → −0.23. Tức hệ số gộp chủ yếu phản ánh khác biệt giữa các nhóm (sigmoid không BN: TD rất âm, acc = 10%) và dấu đổi trong nhóm; 3 hạt giống/cấu hình không độc lập. Một con số sẽ buộc văn bản nói quá, nên theo D2 tôi **bỏ phép tính khỏi `analysis_decay.py` và `results_decay.md`** và dùng mô tả định tính (mục (c): "đi cùng", "đây là mô tả, không phải kiểm định thống kê"; Hạn chế ý (6)). Không có p-value, không có "có ý nghĩa thống kê". Các số trên là từ lần thử trước khi bỏ (log đã xoá cùng script), không còn tái lập được từ file hiện tại.

## 6. Số trang cuối, trang có float mới
38 (cuối Đợt 2) → **40** (<42). Float mới/di chuyển: Bảng 7 `tab:total-degradation` ở p34; Bảng 6 + Hình 3 chung p30 (đã di chuyển); p29 chứa lead-in/định nghĩa mới. Mục Hạn chế + đoạn Adam nằm cuối mục VI (p33–35).

## 7. Điểm tự quyết khác prompt / chưa xác minh
1. **Hạn chế ý (4)** sửa theo repo: spec ghi "chỉ đo tại các epoch ghi trong bảng", nhưng probe được đo ở epoch 0 và sau MỖI epoch (JSON có epoch 0–10); các bảng chỉ ghi epoch 0,1,5,10. Văn bản nêu đúng như vậy.
2. **Layout**: heatmap `[H]` + `width=0.82\linewidth` (thu nhỏ khi nhúng, PNG không đổi) để tránh phao đứng một mình; Bảng 6 vẫn `[htbp]`. Có ~20% trống cuối p29. Nếu Lead muốn giữ `width=\linewidth`, phải chấp nhận Hình 3 đứng riêng một trang.
3. **Spearman bỏ** (mục 5). **Không trích Kingma & Ba** (không thêm bib).
4. Ký hiệu eps của Adam viết `ε_A` để không nhầm với `ε_32`, `ε_64` ở các mục khác.
5. Thêm `\emph{(diễn giải lý thuyết)}` vào "Cơ chế" của (a)(b)(c) — ngoài danh sách D1, nhưng cùng tinh thần.
6. Mục "Hạn chế" đặt cuối 06 (sau notebox của phần training dynamics), trước mục Tổng kết.
7. Chưa xác minh: căn nguyên "There were undefined references" (theo E3, không làm thêm); quyết định mục Hạn chế ý (6) "không kiểm định thống kê" là đúng với repo (không có kiểm định nào được chạy).
8. Dừng; chờ Lead nghiệm thu.
