# TASK_DISPATCH — ĐỢT 3 (ĐỢT CUỐI) (do module4-leader ghi; User đã chỉ đạo nội dung)

> **Coder:** đọc toàn bộ file này, làm ĐÚNG Đợt 3 ghi ở đây, rồi ghi kết quả vào `TASK_REPORT.md`
> theo mục G. Đợt 1 và 2 đã xong (prompt cũ: `TASK_DISPATCH_DOT1.md`, `TASK_DISPATCH_DOT2.md`; chỉ để tham
> chiếu, KHÔNG làm lại). KHÔNG sửa file này.

## Kết quả nghiệm thu Đợt 2 (Lead)

Lead TỰ XÁC MINH: `git diff --stat` trên `results.md`, `results_training.*`, `experiment.py`,
`experiment_training_dynamics.py`, `experiment_numpy_legacy.py`, `figures`, `build.sh` rỗng; `main.log`:
0 Overfull, 0 cảnh báo Reference/Citation undefined, PDF 38 trang và không có `??`/`[?]`; bảng
`tab:jacobian-spectrum` khớp `results_jacobian.md` (đọc từng hàng); các chuỗi `mean ± std` của mục (d)
tồn tại trong `results_training.md`; hai ô epoch 0 không còn `\mathbf`; chứng minh Mệnh đề (3 bước) đã đọc
đầy đủ và đúng; đoạn BatchNorm ở 07 và mục `subsec:dyniso` đã đọc. Lead CHỈ DỰA VÀO Coder cho: nội dung
tóm tắt arXiv của Yang/Xiao (lấy bằng API), kết quả các lần chạy lại `experiment_jacobian.py`
(`cmp`), thử nghiệm bản sao về biblatex. **Căn nguyên** của dòng "There were undefined references"
CHƯA được xác minh; chỉ xác nhận vô hại (không có cảnh báo thật, aux/bbl đủ 9 khoá). Đợt 2 đạt.
User đã đồng ý 3 điều chỉnh của Coder (thu hẹp C4(v) vào nhóm ReLU + Kaiming ở L=32; chỉ bám tóm tắt
Yang/Xiao; "ba lối thoát chính").

```text
VAI TRÒ: Bạn là Coder của dự án báo cáo "Gradient Vanishing" (Module 04). Lead đã rà soát và User đã
duyệt. Bạn chỉ thực hiện ĐỢT 3 dưới đây. Giữ nguyên TIẾNG VIỆT cho văn bản. Không commit/add/push.

THƯ MỤC LÀM VIỆC
  /home/lucius/Project/Learning-AI-engineer/docs/Module-04/gradient-vanishing/
  Interpreter có torch: ../../../code-test/.venv/bin/python (script mới chỉ cần numpy + json).

BƯỚC 0. Chạy `cp TASK_REPORT.md TASK_REPORT_DOT2.md` trước khi ghi báo cáo mới.

QUYỀN FILE
  ĐƯỢC SỬA:  main.tex (chỉ khi cần, nêu lý do), sections/*.tex, references.bib.
  ĐƯỢC TẠO:  analysis_decay.py, results_decay.md, TASK_REPORT_DOT2.md, TASK_REPORT.md (ghi đè).
  CẤM SỬA:   results.md, results_training.json, results_training.md, results_jacobian.md, experiment.py,
             experiment_training_dynamics.py, experiment_jacobian.py, experiment_numpy_legacy.py, figures/*,
             build.sh, TASK_DISPATCH*.md, FEEDBACK_TA_WEEK03.md.
  CẤM: chạy lại bất kỳ thí nghiệm/huấn luyện nào (không chạy experiment.py, experiment_training_dynamics.py,
  không thêm run Adam hay LR sweep). Mọi số mới phải tính từ results_training.json / results.md /
  results_jacobian.md đã có.

TIÊU CHÍ TA CẦN ĐÁP ỨNG: 1.2 (chỉ số chính/phụ), 1.5 và 3 (hạn chế, tương quan), 1.3 (Adam, lý thuyết).

----------------------------------------------------------------------------------------------
A. CHỈ SỐ CHÍNH VÀ PHỤ (script mới analysis_decay.py -> results_decay.md)
----------------------------------------------------------------------------------------------
A1. Định nghĩa trong văn bản (mục "Phép đo" của 06, một chỗ duy nhất, các nơi khác chỉ trỏ về):
  - Chỉ số chính "Tổng suy giảm toàn mạng" (Total Degradation) TD = log10(||g_first||/||g_last||). Nêu
    rõ: TD CHÍNH LÀ log10 rho đã dùng ở các bảng (g_first = tầng đầu/stem, g_last = tầng cuối/head); đặt
    tên TD để khớp rubric; TD < 0: suy giảm về phía tầng đầu, TD > 0: khuếch đại. Không đổi số nào
    ở các bảng cũ; chỉ thêm đoạn định nghĩa/ánh xạ tên.
  - Chỉ số phụ decay_slope = hệ số góc của hồi quy bình phương tối thiểu log10(||g^[l]||) theo chỉ số
    tầng l, tính trên MỌI phần tử của danh sách gnorm (stem, các khối, head: L+2 phần tử). Đơn vị: số bậc
    độ lớn mỗi tầng. Kèm R^2 của chính hồi quy đó. Ghi rõ dấu: slope > 0 nghĩa gradient lớn dần về phía
    head (tức nhỏ dần về phía tầng đầu = vanishing).
A2. analysis_decay.py: đọc results_training.json (trường runs[i].epochs[e].gnorm; arch, L, act, bn, seed,
  diverged). Với mỗi (arch, L, act, bn, epoch in {0, 1, 5, 10}): tính slope và R^2 cho từng seed rồi
  mean ± std (ddof=1) qua seed, cùng định dạng 3 chữ số có nghĩa như results_training.md; bỏ run
  diverged và ghi (n=k) như file nguồn. Nếu có gnorm = 0 hoặc không hữu hạn: KHÔNG lấy log của 0; loại
  phần tử đó khỏi hồi quy và ghi số phần tử bị loại ở cột riêng (hoặc "n/a" kèm lý do nếu còn < 3 điểm).
  Đưa thêm TD (= log10_rho có sẵn trong JSON) vào cùng bảng để đối chiếu với results_training.md (phải
  khớp từng ô; thêm assert so với giá trị tính lại từ gnorm[0]/gnorm[-1]).
  Header của results_decay.md ghi: nguồn là results_training.json, không huấn luyện mới, định nghĩa slope/R^2.
  Chạy hai lần, `cmp` hai file giống nhau.
A3. Đưa vào 06 MỘT bảng gọn `tab:total-degradation` (\captab, caption trên, tự đủ nghĩa, mean ± std, không
  in đậm): ưu tiên L=32, các biến thể ReLU+Kaiming và Sigmoid+Xavier có/không BN, plain/resnet, epoch 0
  và 10, cột TD, decay_slope, R^2. Nếu không vừa một bảng gọn thì đưa bảng đầy đủ vào Phụ lục
  (08-phuluc.tex) và để 06 chỉ một bảng con. Có ít nhất một \ref đứng trước mỗi bảng. KHÔNG tạo hình
  mới; không lặp lại dữ liệu bảng cũ thành hình.
A4. Viết 1 đoạn nhận xét: R^2 thấp ở đâu và vì sao (gradient không đơn điệu theo tầng, ví dụ stem lệch
  mạnh ở PlainNet+BN) nên decay_slope chỉ là xu hướng tuyến tính tổng quát, không thay thế TD. Chỉ mô tả
  số liệu, không đưa cơ chế mới.

----------------------------------------------------------------------------------------------
B. MỤC "HẠN CHẾ CỦA NGHIÊN CỨU" (thêm \subsection cuối 06 hoặc đầu 07, có \label)
----------------------------------------------------------------------------------------------
Liệt kê rõ, mỗi ý 1-2 câu, đúng số liệu repo:
  1. Chỉ dùng SGD thuần (momentum 0, weight decay 0), tốc độ học eta = 0.1 CỐ ĐỊNH cho mọi cấu hình.
     Hệ quả: kết luận "đóng băng/phân kỳ" phụ thuộc eta (r = eta*||g||/||W||); chưa tách được ảnh
     hưởng của kiến trúc khỏi ảnh hưởng của eta. Khảo sát quét LR (ví dụ eta in {0.01, 0.03, 0.1}) là
     HƯỚNG MỞ NGOÀI PHẠM VI, không thực hiện và không có số liệu.
  2. 3 hạt giống (Thí nghiệm 2) và 5 hạt giống (Thí nghiệm 1); khoảng ± rộng; 10/72 run phân kỳ nên
     một số ô chỉ (n=k).
  3. 10 epoch, CIFAR-10 con 10,000 ảnh, MLP rộng 256 (T.N.2) / 64 (T.N.1); không kết luận cho CNN,
     Transformer, tập dữ liệu đầy đủ hay độ sâu khác {8,16,32}.
  4. Chỉ đo tại các epoch ghi trong bảng, trên một probe batch 512; không đo phương sai tiến (forward)
     trực tiếp nên các giải thích cơ chế ở (a)-(d) là diễn giải, không phải đo.
  5. Không thí nghiệm với Adam/AdamW và với khởi tạo trực giao.
  6. Không kiểm định thống kê (chỉ mean ± std trên số hạt giống nhỏ).
Chỉ dùng những ý đúng với repo; nếu một ý sai so với file nguồn thì sửa theo file nguồn và báo lại.

----------------------------------------------------------------------------------------------
C. ĐOẠN LÝ THUYẾT VỀ ADAM (1 \paragraph trong mục Hạn chế, không phải mục riêng)
----------------------------------------------------------------------------------------------
Tiêu đề đoạn phải chứa "thảo luận lý thuyết, không phải quan sát". Nội dung:
  - Cập nhật Adam: Delta theta = - eta * m_hat / (sqrt(v_hat) + eps), eps = 1e-8 (m_hat, v_hat là ước
    lượng moment bậc một/hai đã hiệu chỉnh độ lệch). Viết bằng công thức hiển thị.
  - Phép so sánh phải THEO TỪNG PHẦN TỬ: với mỗi toạ độ i, so sqrt(v_hat_i) với eps. Khi
    sqrt(v_hat_i) >> eps thì bước ~ eta * m_hat_i / sqrt(v_hat_i), gần như không phụ thuộc độ lớn gradient;
    khi sqrt(v_hat_i) cỡ hoặc nhỏ hơn eps thì eps chiếm ưu thế ở mẫu số và bước co lại theo m_hat_i/eps.
    Tuyệt đối KHÔNG dùng chuẩn Frobenius của gradient để so với eps (cập nhật Adam là theo từng phần tử).
  - Hệ quả diễn đạt có điều kiện: "nếu gradient của một số toạ độ ở tầng đầu có độ lớn cỡ eps hoặc nhỏ
    hơn, thì ...". Không khái quát tiêu cực ("Adam không giải quyết vanishing"); không khẳng định Adam
    tốt hay xấu hơn SGD ở đây.
  - Câu bắt buộc: đây là thảo luận lý thuyết, tài liệu không chạy Adam nên không có quan sát nào kèm
    theo; KHÔNG trích bất kỳ con số Adam nào (cấm 0.858, 1.12, 0.47 và mọi số tương tự từ nơi khác).
  - Không thêm trích dẫn mới nếu không tự kiểm chứng được. Nếu muốn dẫn Kingma & Ba 2015 (arXiv
    1412.6980, ICLR 2015) phải có entry bib kiểm chứng và có \cite; nếu không chắc thì KHÔNG thêm.

----------------------------------------------------------------------------------------------
D. LÀM DỊU PHÁT BIỂU CƠ CHẾ QUÁ MẠNH VÀ TƯƠNG QUAN
----------------------------------------------------------------------------------------------
D1. Rà soát sections/06, 07 (và 03 nếu có) các cụm khẳng định nhân quả không được thí nghiệm cô lập.
  Đã biết cần xem: 06 dòng ~157 ("đó chính là hai yếu tố (a) bão hoà và (b) lệch hướng kỳ dị"),
  ~254 ("Đây là lý do một mạng He ... không đạt dynamical isometry"), ~276, ~537 ("nguồn của khoảng cách
  9 điểm phần trăm"), cùng mọi chỗ "là lý do", "chính là", "gây ra", "chứng tỏ", "quyết định". Với mỗi
  chỗ: nếu số liệu chỉ cho thấy sự đi cùng nhau thì đổi sang "phù hợp với", "đi cùng", "gợi ý"; giữ
  nguyên nếu đã chứng minh (ví dụ rank(J) <= rank(D^[1]) là toán, giữ). Ghi trong báo cáo bảng "dòng
  cũ -> dòng mới" cho mọi chỗ sửa, và danh sách chỗ đã rà mà giữ nguyên + lý do.
D2. Tương quan: với các phát biểu "liên quan/đi cùng": thử tính trong analysis_decay.py một Spearman giữa
  TD tại epoch 0 và độ chính xác test epoch 10, trên các run không phân kỳ. Ghi rõ n và rằng 3 seed
  trong cùng cấu hình KHÔNG độc lập; chỉ báo hệ số hạng rho_s và n, KHÔNG báo p-value, KHÔNG viết "có ý
  nghĩa thống kê". Nếu n < 12 hoặc kết quả khiến văn bản phải nói quá, thì BỎ phép tính và đổi phát biểu
  thành mô tả định tính. Dù chọn cách nào, văn bản nói rõ đây là mô tả, không phải kiểm định.
D3. Không sửa số trong bảng cũ; không thêm kết luận mới ngoài những gì số liệu cho thấy.

----------------------------------------------------------------------------------------------
E. MANG SANG
----------------------------------------------------------------------------------------------
E0. Giữ build sạch (D5): 0 Overfull, 0 "!" lỗi, không Reference/Citation undefined. Nếu bảng/đoạn mới
  gây Overfull, sửa cục bộ như Đợt 2 (emergencystretch), không đổi preamble.
E1. Không float mồ côi: không bảng/hình nào đứng một mình trên trang (kiểm bằng `pdftotext -f N -l N`
  hoặc xem ảnh từng trang có float mới/di chuyển). Nếu có, dùng [H] hoặc đổi vị trí lead-in.
E2. Nêu số trang cuối (hiện 38); cố giữ dưới 42.
E3. Căn nguyên "undefined references": KHÔNG làm thêm; giữ ghi chú "vô hại, căn nguyên chưa xác minh".

----------------------------------------------------------------------------------------------
F. NGHIỆM THU (chạy từng lệnh, ghi PASS/FAIL kèm lệnh và output rút gọn)
----------------------------------------------------------------------------------------------
F1. Build: `bash build.sh` rc=0; `grep -a -c Overfull main.log` = 0; `grep -c '^!' main.log` = 0;
    `grep -a -c "Reference .* undefined\|Citation .* undefined" main.log` = 0;
    `pdftotext main.pdf - | grep -c '??\|\[?\]'` = 0.
F2. Nhãn: mọi \label{fig:|tab:} (kể cả tab:total-degradation) có >= 1 \ref đứng trước trong thứ tự trang;
    `grep -nE '(Bảng|Hình)[ ~]*[0-9]' sections/*.tex` rỗng; bảng caption trên, hình caption dưới.
F3. Không float mồ côi (E1): liệt kê từng trang chứa float mới/di chuyển và kết luận.
F4. Số khớp: mỗi ô số của bảng TD/slope/R^2 trong .tex khớp results_decay.md (đối chiếu bằng script,
    in số ô khớp/tổng); cột TD khớp results_training.md; assert trong analysis_decay.py đều OK;
    chạy 2 lần, `cmp results_decay.md` giống nhau.
F5. File cấm không đổi: `git diff --stat -- results.md results_training.json results_training.md
    results_jacobian.md experiment.py experiment_training_dynamics.py experiment_jacobian.py
    experiment_numpy_legacy.py figures build.sh` rỗng. Không có commit.
F6. Hạn chế: `grep -n 'Hạn chế' sections/*.tex` có đúng 1 \subsection; đủ 6 ý (hoặc ý đã sửa theo file
    nguồn, có ghi báo cáo); có câu nêu quét LR là hướng mở ngoài phạm vi.
F7. Adam: `grep -n -i 'adam' sections/*.tex` chỉ nằm trong mục Hạn chế (hoặc nhắc ngắn có trỏ về);
    `grep -n '0\.858\|1\.12\|0\.47' sections/*.tex` không có số Adam; công thức đúng dạng
    m_hat/(sqrt(v_hat)+eps), eps=1e-8; có cụm "từng phần tử" và "không phải quan sát"; không dùng
    "Frobenius" trong đoạn Adam ngoài câu nói KHÔNG dùng nó.
F8. Làm dịu: bảng "dòng cũ -> dòng mới" đầy đủ; `grep -n 'chính là\|là lý do\|nguồn của\|chứng tỏ' sections/06*.tex`
    mỗi kết quả còn lại có ghi chú "đã chứng minh/định nghĩa" trong báo cáo; trích dẫn Yang/Xiao vẫn như Đợt 2.

----------------------------------------------------------------------------------------------
G. BÁO CÁO (ghi đè TASK_REPORT.md; tiếng Việt)
----------------------------------------------------------------------------------------------
  1. Bảng file đã sửa/tạo. 2. F1-F8 PASS/FAIL kèm lệnh + output rút gọn. 3. Đối chiếu số (F4) + trích
  bảng chính của results_decay.md. 4. Bảng "dòng cũ -> dòng mới" (D1) và các chỗ giữ nguyên. 5. Quyết
  định về Spearman (D2): đã tính hay bỏ, vì sao, (rho_s, n) nếu có. 6. Số trang cuối, trang có float
  mới. 7. Mọi điểm tự quyết khác prompt, mọi điều không làm được / chưa xác minh.
  Dừng sau báo cáo; chờ Lead nghiệm thu.
```
