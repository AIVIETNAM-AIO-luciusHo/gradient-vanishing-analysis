# TASK_DISPATCH — ĐỢT 1 (do module4-leader ghi, User đã duyệt)

> **Coder:** đọc toàn bộ file này, làm ĐÚNG đợt ghi ở đây, rồi ghi kết quả vào
> `TASK_REPORT.md` (cùng thư mục) theo định dạng ở mục E. KHÔNG sửa file này.

```text
VAI TRÒ: Bạn là Coder của dự án báo cáo "Gradient Vanishing" (Module 04). Lead (tab
module4-leader) đã rà soát và User đã phê duyệt. Bạn chỉ thực hiện đúng ĐỢT 1 dưới đây.
Giữ nguyên TIẾNG VIỆT cho toàn bộ nội dung văn bản.

THƯ MỤC LÀM VIỆC
  /home/lucius/Project/Learning-AI-engineer/docs/Module-04/gradient-vanishing/
  (có .git riêng; thư mục cha KHÔNG phải repo). Mọi lệnh chạy từ thư mục này.

MỤC TIÊU ĐỢT 1
  Chỉ sửa TRÌNH BÀY theo hướng dẫn của TA (FEEDBACK_TA_WEEK03.md, Phần 1.1 và Tiêu chí 1)
  và thêm 2 mục vào references.bib. KHÔNG đổi số liệu, KHÔNG đổi kết luận khoa học,
  KHÔNG viết thêm nội dung lý thuyết mới (việc đó thuộc Đợt 2 và 3).

---------------------------------------------------------------------
A. PHẠM VI CHO PHÉP / CẤM
---------------------------------------------------------------------
ĐƯỢC SỬA / GHI:
  - main.tex
  - sections/*.tex   (được tạo MỚI sections/08-phuluc.tex)
  - references.bib
  - TASK_REPORT.md   (tạo mới; báo cáo cuối, mục E)
ĐƯỢC CHẠY: bash build.sh; grep/sed -n/cat/ls/python3 -I (chỉ đọc); sao chép thư mục ra
  chỗ tạm NGOÀI repo để thử nghiệm (xem mục D6).
CẤM TUYỆT ĐỐI:
  - Sửa results_training.json, results_training.md, results.md, results_*.* bất kỳ;
    experiment*.py, *.py.bak, *.log, *.txt; build.sh; .gitignore; *.sty;
    FEEDBACK_TA_WEEK03.md; TASK_DISPATCH.md; figures/*.png; Figures/*; README.md.
  - git commit / git add / git push / git checkout / git reset. KHÔNG commit.
  - Chạy lại bất kỳ experiment*.py nào, hoặc vẽ lại PNG.
  - Trích các số Adam của TA (0.858; decay_slope 1.12->0.47): chúng KHÔNG thuộc repo này.
  - Đoán/bịa thông tin thư mục (bib, số liệu, trang).

---------------------------------------------------------------------
B. CÁC THAY ĐỔI (số dòng là TRƯỚC khi sửa; dòng sẽ dịch chuyển, hãy định vị bằng nội dung)
---------------------------------------------------------------------

B1. Macro caption bảng mới (main.tex)
  - main.tex:126 định nghĩa tcolorbox `captionbelowboxaivn` (tên sai nghĩa vì caption bảng
    phải nằm TRÊN). Đổi tên thành `captionaboveboxaivn` và cập nhật mọi chỗ dùng (hiện có
    tại 04-vidu.tex:226, 06-thucnghiem.tex:109,135,334, 07-tongket.tex:60).
  - Thêm counter `bang` và macro `\captab`, tương tự `\capfig` ở main.tex:157-163:
        \newcounter{bang}
        \newcommand{\captab}[1]{\refstepcounter{bang}Bảng~\thebang:~#1}
    Nhãn đặt BÊN TRONG đối số, sau văn bản, giống cách `\capfig{...\label{fig:x}}` đang làm.
  - Quy ước vị trí: trong mỗi môi trường `table`, hộp caption đứng TRƯỚC `\begin{tabular}`
    (hoặc trước `\begin{adjustbox}` ở 06-thucnghiem.tex:302). Ở bảng 06:298-340, phần
    chú thích `^*`/`^†` (minipage 326-332) vẫn nằm dưới bảng.
  - Thêm khoảng cách hợp lý giữa hộp caption và bảng (vd. \medskip hoặc `after skip`);
    nhìn PDF để chỉnh cho đẹp.

B2. Bỏ số bảng gõ tay, chuyển sang \captab + \label + \ref
  Các điểm cụ thể:
  - 03-nguyennhan.tex:51-66 (bảng 0.25^L): hiện KHÔNG có caption. Thêm caption phía trên
    với nhãn tab:suy-giam-luy-thua, nêu: đại lượng 0.25^L theo số tầng L, và câu phát hiện
    (L=10 chỉ còn khoảng 8 lần epsilon máy float32; L=20 dưới epsilon máy). Đổi câu dẫn
    ở dòng 49 ("Bảng suy giảm dưới đây cho thấy...") thành có \ref tới bảng này.
  - 04-vidu.tex:210-231: bảng cross-entropy, đang là "Bảng 2:" gõ tay (dòng 227). Đổi sang
    \captab, đặt caption trên, nhãn tab:loss-compare. Thêm \ref vào câu dẫn trước bảng
    (dòng 205-208, câu "đúng bằng tỉ lệ đã tính ở cả hai cột của bảng dưới").
  - 06-thucnghiem.tex:89-113: đang là "Bảng 3:" gõ tay (dòng 110). Nhãn tab:grad-init.
  - 06-thucnghiem.tex:115-139: đang là "Bảng 4:" gõ tay (dòng 136). Nhãn tab:freeze.
  - 06-thucnghiem.tex:84-87: câu "Hai bảng dưới chép nguyên từ results.md..." phải dẫn bằng
    \ref tới cả tab:grad-init và tab:freeze.
  - 06-thucnghiem.tex:212 ("số liệu của Bảng~3") và :236 ("như Bảng~4"): thay bằng \ref.
  - 06-thucnghiem.tex:297 `\setcounter{table}{4}` và 335-336 `\refstepcounter{table}...
    \thetable`: XOÁ cơ chế hack này; bảng này (nhãn tab:training-dynamics, GIỮ NGUYÊN nhãn)
    dùng \captab. Xoá luôn comment giải thích hack ở 295-296.
  - 07-tongket.tex:16-64: đang là "Bảng 6:" gõ tay (dòng 61). Nhãn tab:nguyen-ly-khac-phuc.
    Thêm \ref tới bảng này trong đoạn mở đầu mục (trước dòng 16).
  Kỳ vọng: sau khi sửa, số bảng tự sinh theo thứ tự xuất hiện sẽ là 1..6, trùng với số cũ.

B3. Caption độc lập (TA: nêu vẽ gì, thiết lập nào gồm mô hình/độ sâu/số seed, + 1 câu phát hiện)
  Viết lại/bổ sung cho các caption sau (giữ ngắn, KHÔNG thêm số liệu mới; chỉ dùng số đã có
  trong văn bản hoặc results_training.md):
  - Bảng grad-init (06:110): thêm thiết lập (MLP rộng 64, N=256, đo tại bước khởi tạo,
    5 hạt giống) + phát hiện (sigmoid+Xavier: log10ρ giảm gần tuyến tính theo L;
    ReLU+He: nằm gần 0).
  - Bảng freeze (06:136): thêm thiết lập (η=0.1) + phát hiện (đóng băng hoàn toàn chỉ ở
    sigmoid+Xavier, L=20, 5/5 hạt giống).
  - Bảng training-dynamics (06:336-338): thêm thiết lập (MLP rộng 256, CIFAR-10 10k ảnh,
    SGD η=0.1, 10 epoch, probe batch 512; 3 hạt giống) + một câu phát hiện đã có trong
    bản văn mục (c) (ResNet+BN vượt PlainNet+BN ở L=32).
  - Bảng nguyen-ly-khac-phuc (07:61-62): thêm một câu "đọc bảng này thế nào"/kết luận chính.
  - Hình heatmap (06:346-352) và hình gradflow (06:360-366): bổ sung "MLP rộng 256,
    CIFAR-10 10k ảnh, SGD η=0.1" nếu còn thiếu; nêu rõ trục giá trị là log10 (đã có hoặc thêm).
  - Hình gradnorm (06:211-214, sẽ chuyển về Phụ lục, xem B5): viết lại độc lập, ví dụ:
    "log10ρ, ρ=‖g^[1]‖_F/‖g^[L]‖_F, theo độ sâu L∈{5,10,20} của MLP rộng 64 tại bước khởi
    tạo, trung bình ± độ lệch chuẩn trên 5 hạt giống; điều kiện A (sigmoid+Xavier) giảm gần
    tuyến tính theo L (khoảng 0.2 mỗi tầng), điều kiện B (ReLU+He) nằm gần 0. Trục y là
    log10ρ (tương đương thang log)." Bỏ cụm "số liệu của Bảng 3" như là phần chính của caption.

B4. Mọi hình/bảng phải được dẫn bằng \ref TRƯỚC khi xuất hiện (thứ tự nguồn)
  Hiện các nhãn sau KHÔNG có \ref hoặc \ref đứng SAU phao: fig:graph (02:155),
  fig:sat (03:98), fig:gradnorm (06:183), fig:heatmap-plain-resnet (06:342, \ref đầu tiên ở
  dòng 392), fig:gradflow-residual (06:356, \ref đầu tiên ở dòng 393/431),
  tab:training-dynamics (06:298, \ref đầu tiên ở dòng 428).
  Cần thêm câu dẫn (1-2 câu, nêu thứ tự xu hướng -> cơ chế, KHÔNG đọc lại từng số) ở:
  - 02-cosotoan.tex quanh dòng 148-153: dẫn Hình~\ref{fig:graph}.
  - 03-nguyennhan.tex quanh dòng 76-90 (đoạn về tanh/sigmoid): dẫn Hình~\ref{fig:sat}.
  - 06-thucnghiem.tex đoạn "Thiết lập"/"Phép đo" (dòng 257-293, TRƯỚC dòng 297):
    dẫn Bảng~\ref{tab:training-dynamics}, Hình~\ref{fig:heatmap-plain-resnet},
    Hình~\ref{fig:gradflow-residual}.
  - 06-thucnghiem.tex đoạn Phân tích (quanh dòng 141-181, trước phụ lục): dẫn
    Hình~\ref{fig:gradnorm} ở Phụ lục (tham chiếu tiến, bằng chữ "ở Phụ lục", vd.
    "xem Hình~\ref{fig:gradnorm} ở Phụ lục~A").
  Không được làm đổi nghĩa khoa học của đoạn văn đã có.

B5. Chuyển fig:gradnorm về Phụ lục (KHÔNG xoá; lý do: trùng dữ liệu với Bảng grad-init)
  - Cắt toàn bộ môi trường figure ở 06-thucnghiem.tex:183-216 (kể cả tikzpicture, caption và
    nhãn fig:gradnorm) sang file mới sections/08-phuluc.tex.
  - Tạo sections/08-phuluc.tex: bắt đầu bằng \clearpage\appendix (CHÚ Ý: main.tex nạp gói
    `appendix` với tùy chọn `page`; KHÔNG dùng môi trường `appendices` vì nó chèn thêm một
    trang tiêu đề "Phụ lục" gần như trống). Sau \appendix, \thesection mặc định bị đặt lại
    thành chữ cái không có dấu chấm trong khi main.tex dùng "\Roman{section}." cho thân bài;
    chỉnh \renewcommand{\thesection}{\Alph{section}.} sao cho tiêu đề phụ lục đọc được
    (vd. "A. Hình bổ sung: log10ρ theo độ sâu tại bước khởi tạo") và số mục con không
    bị dính ("A.1." chứ không phải "A1."). Đặt \label{app:phu-luc-a} cho mục.
  - Trong main.tex, thêm \input{sections/08-phuluc} NGAY SAU \input{sections/07-tongket}
    (trước \printbibliography).
  - Trong 06-thucnghiem.tex, tại vị trí cũ của hình, không để lại khoảng trắng thừa.
  - Phần thân bài sau khi chuyển còn 4 hình (graph, sat, heatmap, gradflow) = trong khoảng
    3-5 mà TA yêu cầu. KHÔNG chuyển thêm hình khác.

B6. Quy cách trình bày thêm (TA Phần 1.1)
  - In đậm giá trị TỐT NHẤT chỉ ở những bảng có khái niệm "tốt nhất" rõ ràng: trong
    Bảng training-dynamics, ở cột Test acc (%), tại mỗi hàng epoch, in đậm giá trị trung bình
    cao hơn giữa PlainNet và ResNet, CHỈ khi cả hai cùng không phân kỳ. KHÔNG in đậm cột ρ,
    gnorm hay các bảng khác (không có "tốt nhất" tự nhiên). Báo lại nếu bạn thấy nơi khác
    cũng cần.
  - Hình gradnorm (TikZ/pgfplots): màu A=aivnGreen, B=brandIndigo đã đúng, giữ nguyên.
    Trục y là log10ρ (đại lượng đã log), không đổi sang \ymode=log.
  - Hai PNG (figures/heatmap_plain_vs_resnet.png, figures/gradient_flow_residual.png):
    QUYẾT ĐỊNH CỦA USER: GIỮ NGUYÊN, không sửa code vẽ, không chạy lại, để đảm bảo an toàn
    dữ liệu. Lead đã đọc script sinh hình (experiment_training_dynamics.py:282-333): cả hai
    vẽ giá trị log10 trên trục tuyến tính (tương đương thang log); gradient_flow_residual
    dùng bảng màu mặc định của matplotlib theo (arch, L), chưa cố định màu theo phương
    pháp. Chỉ ghi nhận trong báo cáo cuối như một hạn chế đã biết, KHÔNG xử lý.

B7. references.bib: thêm 2 mục (KHÔNG cite chúng ở Đợt 1; việc cite thuộc Spec #1)
  Theo đúng kiểu định dạng sẵn có trong file (@inproceedings, booktitle ghi đầy đủ + viết tắt):
  - Khoá: Yang2019MeanField
      author    = Yang, Greg and Pennington, Jeffrey and Rao, Vinay and
                  Sohl-Dickstein, Jascha and Schoenholz, Samuel S.
      title     = A Mean Field Theory of Batch Normalization
      booktitle = International Conference on Learning Representations (ICLR)
      year      = 2019
  - Khoá: Xiao2018DynIso
      author    = Xiao, Lechao and Bahri, Yasaman and Sohl-Dickstein, Jascha and
                  Schoenholz, Samuel S. and Pennington, Jeffrey
      title     = Dynamical Isometry and a Mean Field Theory of {CNN}s: How to
                  Train 10,000-Layer Vanilla Convolutional Neural Networks
      booktitle = Proceedings of the 35th International Conference on Machine
                  Learning (ICML)
      year      = 2018
  YÊU CẦU KIỂM CHỨNG: Lead nhớ các thông tin trên nhưng CHƯA tra lại nguồn. Nếu có thể tra
  (arXiv/OpenReview/PMLR), hãy đối chiếu tên tác giả, tiêu đề, hội nghị, năm. Với trường
  nào bạn KHÔNG chắc, để TRỐNG (bỏ trường) và báo lại. KHÔNG tự điền số trang, volume,
  URL hay DOI nếu không kiểm chứng được. KHÔNG sửa 7 mục bib cũ.

B8. Overfull ở mục lục (QUYẾT ĐỊNH CỦA USER: Coder tự xử lý trong main.tex)
  - main.log hiện có 1 cảnh báo `Overfull \hbox (2.63356pt too wide)` ở trang 1, dòng "VII."
    của mục lục. Tăng \cftsecnumwidth tại main.tex:73 (hiện 3em; thử 3.5em hoặc 4em, chọn
    giá trị nhỏ nhất làm hết cảnh báo) và kiểm tra các cảnh báo kia không xuất hiện lại.
  - Nếu sau khi chỉnh vẫn còn Overfull > 1pt, báo nguyên nhân trong TASK_REPORT.md.

---------------------------------------------------------------------
C. KHÔNG LÀM Ở ĐỢT 1
---------------------------------------------------------------------
  - Không cite Yang/Xiao trong văn bản; không viết mục Dynamical Isometry (Đợt 2 = Spec #1).
  - Không viết mục Hạn chế, không thêm R²/decay_slope, không thảo luận Adam (Đợt 3 = Spec #2).
  - Không đổi nội dung khoa học của 07-tongket.tex:89-96 (sẽ viết lại ở Đợt 2).

---------------------------------------------------------------------
D. NGHIỆM THU (chạy đúng các lệnh này, báo pass/fail kèm lệnh và output rút gọn)
---------------------------------------------------------------------
D1. Mỗi nhãn fig:/tab: có ít nhất một \ref đứng TRƯỚC \label theo thứ tự đọc:
    python3 -I - <<'EOF'
    import re, glob
    files = sorted(glob.glob('sections/0[1-8]-*.tex'))
    txt = ''.join(open(f, encoding='utf-8').read() for f in files)
    for lab in re.findall(r'\\label\{((?:fig|tab):[^}]+)\}', txt):
        lp = txt.index('\\label{%s}' % lab)
        rp = [m.start() for m in re.finditer(r'\\(?:ref|cref|autoref)\{%s\}' % re.escape(lab), txt)]
        print(lab, 'PASS' if rp and min(rp) < lp else 'FAIL')
    EOF
    (Thứ tự file 01..08 khớp thứ tự \input trong main.tex. Mọi dòng phải PASS.)
D2. Không còn số bảng/hình gõ tay:
    grep -nE '(Bảng|Hình)[ ~]*[0-9]' main.tex sections/*.tex      -> phải RỖNG
    grep -nE 'setcounter\{table\}|\\thetable|captionbelowboxaivn' main.tex sections/*.tex -> RỖNG
D3. Vị trí caption: trong mọi env `table`, `captionaboveboxaivn` đứng trước `\begin{tabular}`;
    trong mọi env `figure`, `figurecaptionmodern` đứng sau hình. Kiểm bằng
    grep -nE 'begin\{(table|figure)\}|captionaboveboxaivn|figurecaptionmodern|begin\{tabular\}|includegraphics|end\{tikzpicture\}' sections/*.tex
    rồi trình bày bảng thứ tự cho từng bảng/hình.
D4. fig:gradnorm nằm trong sections/08-phuluc.tex và không còn trong 06-thucnghiem.tex:
    grep -n 'fig:gradnorm' sections/*.tex ; kiểm tra thêm PDF không có trang "Phụ lục" trống.
D5. Build: chạy `bash build.sh` (3 lượt pdflatex + 1 bibtex, tuần tự, KHÔNG chạy song song).
    Sau đó:
    grep -ciE 'undefined|multiply defined' main.log                -> phải = 0
    grep -nE 'Overfull' main.log                                    -> không dòng nào > 1pt
    (xem B8 về cảnh báo Overfull có sẵn ở mục lục.)
    Các cảnh báo CÓ SẴN và ĐƯỢC PHÉP: biblatex "csquotes missing", "Language 'vietnamese' not
    supported", pgfplots compatibility mode, LaTeX Font Warning cỡ 5.5. (Có thể nạp
    `csquotes` nếu muốn, tuỳ chọn.)
    Không có "LaTeX Warning: Reference ... undefined" / "Citation ... undefined".
D6. Entry bib mới không gây cảnh báo BibTeX. Vì chúng chưa được cite nên bibtex sẽ bỏ qua;
    để kiểm tra, sao chép thư mục ra chỗ tạm NGOÀI repo (vd. thư mục scratchpad hoặc
    $(mktemp -d)), thêm `\nocite{Yang2019MeanField,Xiao2018DynIso}` ở BẢN SAO, build,
    rồi kiểm `grep -n 'Warning' main.blg` (phải rỗng cho hai khoá này) và xác nhận hai mục
    xuất hiện đúng trong danh sách tài liệu tham khảo. KHÔNG để `\nocite` trong repo.
D7. Số trang: ghi số trang PDF trước (32 trang theo ghi chú) và sau; giải thích nếu chênh lệch
    đáng kể. Mở/đọc PDF (pdftotext hoặc xem ảnh) để xác nhận: bảng đánh số 1..6 đúng thứ tự,
    hình thân bài đánh số 1..4 rồi Hình 5 ở Phụ lục, không có hình/bảng mồ côi, không có bảng
    nào bị tràn lề.
D8. Số liệu không đổi: so sánh các con số trong bảng của 06-thucnghiem.tex trước và sau
    (git diff -- sections/06-thucnghiem.tex), chỉ phần caption/dẫn/bố cục được thay đổi, KHÔNG
    ô số nào đổi. `git status --short` phải cho thấy chỉ: main.tex, sections/*.tex
    (kể cả 08-phuluc.tex mới), references.bib, TASK_DISPATCH.md và TASK_REPORT.md (untracked),
    và các file build (main.pdf, và các file đã bị gitignore); results_*, experiment*.py,
    figures/ phải KHÔNG xuất hiện.

---------------------------------------------------------------------
E. ĐỊNH DẠNG BÁO CÁO CUỐI — GHI VÀO TASK_REPORT.md
---------------------------------------------------------------------
  1. Danh sách file đã sửa/tạo (kèm tóm tắt 1 dòng mỗi file).
  2. Bảng nghiệm thu D1..D8: Pass/Fail, LỆNH đã chạy, output rút gọn.
  3. Số trang PDF trước/sau.
  4. Cách xử lý B5 (Phụ lục): đánh số mục, có trang trống hay không.
  5. Kết quả kiểm chứng bib B7: trường nào đã đối chiếu với nguồn nào, trường nào để trống.
  6. Kết quả B8 (giá trị \cftsecnumwidth cuối cùng, Overfull còn lại hay không).
  7. Mọi điểm KHÔNG làm được hoặc tự quyết định khác với prompt (kèm lý do), và mọi điểm
     cần Lead/User quyết.
  8. Không commit. Dừng sau khi ghi TASK_REPORT.md; chờ Lead đánh giá trước khi sang Đợt 2.
```
