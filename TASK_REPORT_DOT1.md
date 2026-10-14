# TASK_REPORT — ĐỢT 1 (Coder → Lead)

Không commit, không chạy lại experiment, không sửa file cấm. Build cuối: `bash build.sh` exit 0.

## 1. File đã sửa / tạo

| File | Tóm tắt |
|---|---|
| `main.tex` | Đổi `captionbelowboxaivn`→`captionaboveboxaivn` (+`after skip=6pt`); thêm counter `bang` + `\captab`; `\cftsecnumwidth` 3em→3.5em; thêm `\input{sections/08-phuluc}` sau 07. |
| `sections/02-cosotoan.tex` | Thêm câu dẫn `Hình~\ref{fig:graph}` trước hình. |
| `sections/03-nguyennhan.tex` | Bảng 0.25^L có caption trên (`tab:suy-giam-luy-thua`), câu dẫn dùng `\ref`; thêm câu dẫn `Hình~\ref{fig:sat}`. |
| `sections/04-vidu.tex` | Bảng cross-entropy: `\captab` + caption trên, nhãn `tab:loss-compare`, `\ref` ở câu dẫn. |
| `sections/06-thucnghiem.tex` | Bảng grad-init / freeze / training-dynamics chuyển sang `\captab`, caption trên + thiết lập + phát hiện; xoá hack `\setcounter{table}{4}`/`\thetable`; thay "Bảng~3/4" bằng `\ref`; thêm đoạn dẫn Bảng/Hình ở cuối "Phép đo"; dẫn `fig:gradnorm` ở Phụ lục; cắt figure gradnorm; in đậm Test acc tốt nhất; bổ sung thiết lập vào caption heatmap & gradflow. |
| `sections/07-tongket.tex` | Bảng 6: `\captab` + caption trên (`tab:nguyen-ly-khac-phuc`) + câu "đọc bảng thế nào"; `\ref` ở đoạn mở đầu. |
| `sections/08-phuluc.tex` (MỚI) | `\clearpage\appendix`, `\thesection`=`\Alph{section}.`, mục "A. Hình bổ sung…" (`app:phu-luc-a`), chứa Hình gradnorm với caption viết lại độc lập. |
| `references.bib` | Thêm `Yang2019MeanField`, `Xiao2018DynIso` (7 mục cũ không đổi). |
| `main.pdf` (build) | Tracked trong git, bị build ghi lại. |

## 2. Nghiệm thu D1–D8

| # | Kết quả | Lệnh / output rút gọn |
|---|---|---|
| D1 | **PASS** | script `python3 -I` của mục D1: 11 nhãn (fig:graph, tab:suy-giam-luy-thua, fig:sat, tab:loss-compare, tab:grad-init, tab:freeze, tab:training-dynamics, fig:heatmap-plain-resnet, fig:gradflow-residual, tab:nguyen-ly-khac-phuc, fig:gradnorm) đều PASS |
| D2 | **PASS** | cả hai `grep` rỗng |
| D3 | **PASS** | Mọi `table`: caption box → `\begin{tabular}`/`tabularx` (03:53<58, 04:212<218, 06:92<99, 120<127, 275<285, 07:19<26). Mọi `figure`: hình/tikz → `figurecaptionmodern` (02:199<201, 03:140<142, 06:320<321, 334<335, 08:37<39). Chú thích `^*/^†` của bảng training-dynamics vẫn dưới bảng. |
| D4 | **PASS** | `grep -n fig:gradnorm`: 06:151 (`\ref`), 08:8 (`\ref`), 08:45 (`\label`); không còn `\label` trong 06. Không có trang Phụ lục trống (từ-trang: trang 33 có hình + Tài liệu tham khảo). |
| D5 | **PARTIAL (xem mục 7)** | `bash build.sh` rc=0. `grep -ciE 'undefined\|multiply defined' main.log` = 1: chỉ là dòng "There were undefined references" + biblatex "(re)run BibTeX", **giống hệt bản gốc** (baseline build từ `git archive HEAD` cũng có); không có "Reference/Citation … undefined", PDF không có `??`/`[?]`. Overfull: còn 2 (26.47pt, 9.45pt) — đều CÓ SẴN ở baseline, xem mục 7. Overfull mục lục (2.63pt) đã hết. 0 dòng bắt đầu bằng `!`. |
| D6 | **PASS** | Bản sao ngoài repo + `\nocite{Yang2019MeanField,Xiao2018DynIso}`: `grep Warning main.blg` = 0; PDF có `[8] G. Yang et al., "A mean field theory of batch normalization", … ICLR, 2019.` và `[9] L. Xiao et al., "Dynamical isometry … CNNs …", … ICML, 2018, pp. 5393–5402.` Không có `\nocite` trong repo. |
| D7 | **PASS** | Bảng 1…6 theo đúng thứ tự (03, 04, 06, 06, 06, 07); Hình 1–4 thân bài (02, 03, 06, 06), Hình 5 ở Phụ lục A (tr.33). Không hình/bảng mồ côi. Đã xem ảnh trang 26 và 33: không tràn lề. |
| D8 | **PASS** | `git diff` 06: chỉ đổi caption/dẫn/bố cục + 5 ô Test acc bọc `\mathbf{}`; mọi giá trị số giữ nguyên. `git status --short`: M main.pdf, main.tex, references.bib, sections/02,03,04,06,07; ?? sections/08-phuluc.tex, TASK_DISPATCH.md, FEEDBACK_TA_WEEK03.md, experiment_stdout.txt, training_dynamics_sanity.log (4 file untracked cuối có sẵn từ trước). results_*, experiment*.py, figures/ không xuất hiện. |

## 3. Số trang PDF
Trước: 32 → Sau: 33 (+1). Nguyên nhân: caption dài hơn, 5 câu dẫn, tiêu đề Phụ lục, hình gradnorm chuyển sang trang cuối cùng với Tài liệu tham khảo.

## 4. B5 – Phụ lục
`\clearpage\appendix` (không dùng `appendices`), `\renewcommand{\thesection}{\Alph{section}.}` → tiêu đề "A. Hình bổ sung: log10ρ theo độ sâu tại bước khởi tạo"; mục lục ghi "A. … 33". `\ref{app:phu-luc-a}` in "A." (cùng quy ước chấm với "Mục III." hiện có). Không có trang trống. (Không có subsection nên chưa kiểm "A.1.", cấu hình `\thesubsection` kế thừa `\thesection\arabic{subsection}.` ⇒ "A.1.".)

## 5. B7 – Kiểm chứng bib (WebSearch)
- **Yang2019MeanField**: tác giả (5 người, đúng thứ tự), tiêu đề, ICLR 2019 — khớp arXiv 1902.08129 / ICLR poster 802. Không điền pages/URL.
- **Xiao2018DynIso**: tác giả, tiêu đề, ICML 2018 — khớp PMLR v80 / arXiv 1806.05393. **Tự bổ sung `pages = {5393--5402}`** (PMLR 80:5393–5402, từ kết quả tìm kiếm); prompt cho phép điền khi kiểm chứng được. Nếu muốn đúng nguyên văn prompt, xoá dòng này.
- Không trường nào để trống.

## 6. B8
`\cftsecnumwidth` = **3.5em** (giá trị thử đầu tiên; chưa thử 3em<x<3.5em). Overfull mục lục hết.

## 7. Điểm không làm được / tự quyết / cần Lead–User quyết
1. **D5 Overfull > 1pt còn 2, đều có sẵn ở bản gốc (kiểm chứng bằng build `git archive HEAD`)**, nằm trong 06-thucnghiem.tex: (i) đoạn "Phép đo" (26.47pt) — công thức inline `\mathbf{g}_{\text{stem}} = \partial…` và `\mathbf{g}_{\text{head}} = …` không ngắt dòng được; (ii) mục (c) gạch đầu dòng "ρ của ResNet ở gần 1" (9.45pt). Ngoài phạm vi Đợt 1, chưa sửa. Đề xuất: đưa hai công thức thành `\[ \]` hoặc thêm `\allowbreak`/`\sloppy`. Cần Lead quyết.
2. Cảnh báo "There were undefined references" / "(re)run BibTeX" có sẵn ở baseline (chuỗi build.sh + biblatex bibtex-backend), không phải reference thực sự thiếu.
3. B2/04: câu cũ "cả hai cột của bảng dưới" → "cả hai hàng của Bảng~\ref{tab:loss-compare}" (tỉ lệ nằm ở cột cuối của 2 hàng; sửa cho đúng, không đổi nghĩa khoa học).
4. B6 in đậm: áp dụng đúng quy tắc, gồm cả epoch 0 (PlainNet 10.6 và 9.94 ở khối không-BN), dù chênh lệch ở mức đoán ngẫu nhiên và epoch 0 là trước huấn luyện. Cần Lead quyết có bỏ hai ô epoch 0 không. Không thấy bảng nào khác cần in đậm.
5. Hạn chế đã biết (B6, không xử lý): hai PNG vẽ log10 trên trục tuyến tính; `gradient_flow_residual.png` dùng bảng màu mặc định theo (arch, L), chưa cố định màu theo phương pháp.
6. Khoảng cách caption–bảng: `after skip=6pt` trên hộp caption; đã xem ảnh p.26, trông ổn.
7. Dừng sau khi ghi báo cáo; chờ Lead đánh giá trước Đợt 2.
