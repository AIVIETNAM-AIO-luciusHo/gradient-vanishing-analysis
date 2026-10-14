# TASK_DISPATCH — ĐỢT 2 (do module4-leader ghi; User đã chỉ đạo nội dung)

> **Coder:** đọc toàn bộ file này, làm ĐÚNG Đợt 2 ghi ở đây, rồi ghi kết quả vào `TASK_REPORT.md`
> theo mục F. Đợt 1 đã xong và được Lead nghiệm thu (bản prompt Đợt 1 lưu ở
> `TASK_DISPATCH_DOT1.md`, chỉ để tham chiếu; KHÔNG làm lại Đợt 1). KHÔNG sửa file này.

## Kết quả nghiệm thu Đợt 1 (Lead)

Lead TỰ XÁC MINH: D1 (11 nhãn PASS), D2 (grep rỗng), số trang 33, `git status`, `main.tex:73` =
3.5em, `08-phuluc.tex`, nội dung bib, ô in đậm, các hàng số của `06-thucnghiem.tex` không đổi,
`main.log`: còn 2 Overfull (26.47pt và 9.45pt). Lead CHỈ DỰA VÀO báo cáo của Coder cho D3
(thứ tự caption từng bảng/hình), D6 (thử `\nocite` ở bản sao) và nguồn bib (WebSearch: PMLR v80 /
arXiv 1806.05393 cho Xiao; arXiv 1902.08129 / ICLR cho Yang). Đợt 1 đạt, D5 được miễn tạm.

```text
VAI TRÒ: Bạn là Coder của dự án báo cáo "Gradient Vanishing" (Module 04). Lead đã rà soát và User
đã duyệt. Bạn chỉ thực hiện đúng ĐỢT 2 dưới đây. Giữ nguyên TIẾNG VIỆT cho văn bản.

THƯ MỤC LÀM VIỆC
  /home/lucius/Project/Learning-AI-engineer/docs/Module-04/gradient-vanishing/
  (có .git riêng). Mọi lệnh chạy từ thư mục này.

MỤC TIÊU ĐỢT 2
  Đáp ứng Tiêu chí 4 và 5 của TA (FEEDBACK_TA_WEEK03.md, Phần 1.4, 1.5 và Phần 3):
  (i) thêm khung lý thuyết Dynamical Isometry: phổ của TOÀN BỘ Jacobian đầu-cuối, kèm đo đạc;
  (ii) trích Yang et al. 2019 và Xiao et al. 2018 (đã có trong references.bib);
  (iii) viết lại đoạn BatchNorm/He ở 07-tongket.tex cho đúng yêu cầu của TA ("không có gì đảm bảo
  duy trì ổn định"); (iv) xử lý các việc mang sang từ Đợt 1.
  KHÔNG làm ở Đợt này: mục "Hạn chế", R²/decay_slope, quét learning rate, thảo luận Adam (Đợt 3).

---------------------------------------------------------------------
A. PHẠM VI CHO PHÉP / CẤM
---------------------------------------------------------------------
ĐƯỢC SỬA / GHI: main.tex, sections/*.tex, references.bib (chỉ nếu phát hiện lỗi trường; xem F),
  TASK_REPORT.md.
ĐƯỢC TẠO MỚI: experiment_jacobian.py, results_jacobian.md, TASK_REPORT_DOT1.md (bản lưu của
  TASK_REPORT.md Đợt 1: HÃY `cp TASK_REPORT.md TASK_REPORT_DOT1.md` TRƯỚC khi ghi báo cáo mới).
ĐƯỢC CHẠY: bash build.sh; python3 experiment_jacobian.py (rẻ, vài giây đến vài phút, xem B3);
  grep/sed -n/cat/ls (dùng `grep -a` với main.log vì file không phải UTF-8 thuần);
  python3 -I (chỉ đọc); sao chép thư mục ra chỗ tạm NGOÀI repo để thử.
  Interpreter: dùng đúng Python có torch 2.14.0+cpu / Python 3.12.3 như header của results.md
  (kiểm bằng `python3 -c "import torch;print(torch.__version__)"`; nếu khác, báo lại).
CẤM TUYỆT ĐỐI:
  - Sửa experiment.py, experiment_training_dynamics.py, experiment_numpy_legacy.py, *.py.bak,
    results_training.json, results_training.md, results.md, *.log, *.txt, build.sh, .gitignore,
    *.sty, FEEDBACK_TA_WEEK03.md, TASK_DISPATCH*.md, figures/*.png, Figures/*, README.md.
  - Chạy `python experiment.py` (nó ghi đè results.md) hoặc experiment_training_dynamics.py.
    NHƯNG được `from experiment import build, seed_all, SEEDS, DEPTHS, D, N` (main() có guard).
  - git commit / add / push / checkout / reset. KHÔNG commit.
  - Trích số Adam của TA (0.858; decay_slope 1.12->0.47).
  - Bịa tham chiếu, số liệu, hay phát biểu định lý mà bạn không kiểm chứng được.

---------------------------------------------------------------------
B. THÍ NGHIỆM MỚI: experiment_jacobian.py  ->  results_jacobian.md
---------------------------------------------------------------------
B1. Đối tượng đo. Dùng ĐÚNG mạng, khởi tạo và dữ liệu của experiment.py (L tầng tuyến tính, L-1 tầng
    ẩn rộng D=64 + 1 tầng ra 1 chiều, bias=0; A: Sigmoid+xavier_normal_, B: ReLU+kaiming_normal_;
    seeds [42,100,2024,2026,9999]; L∈{5,10,20}; seed_all(seed) -> x,y=randn(N,D),randn(N,1) ->
    build(L, act)). Dùng lại build/seed_all/hằng số bằng import (đừng chép lại logic).

B2. LƯU Ý KỸ THUẬT BẮT BUỘC: tầng ra là 1 chiều, nên Jacobian đầu vào -> đầu ra là vector 1xD
    (hạng 1): phổ suy biến, vô nghĩa. Vì vậy tính J trên CHỒNG TẦNG ẨN 64x64, không tính tới đầu
    ra vô hướng:
        J = ∂z^[L-1] / ∂z^[1]   (z^[k] là tiền kích hoạt tầng k, theo ký hiệu của 02-cosotoan.tex)
          = W^[L-1] D^[L-2] ... W^[2] D^[1],      D^[k] = diag(σ'(z^[k])),   J ∈ R^{64x64},
    gồm L-2 thừa số, đúng eq. (jacforward) của báo cáo. L∈{5,10,20} nên L>=3.
    Tính J bằng torch.autograd.functional.jacobian trên hàm z^[1] -> z^[L-1] cho từng mẫu x_i.
    Dùng float64 CHỈ cho phép đo phổ: khởi tạo mạng ở float32 như experiment.py (để trọng số y hệt
    results.md), rồi .double() cả mạng và dữ liệu. Ghi rõ điều này trong header results_jacobian.md.
    (KHÔNG đặt torch.set_default_dtype(float64) toàn cục nếu nó đổi trọng số khởi tạo.)

B3. Đại lượng báo cáo. Với mỗi (điều kiện, L, seed): lấy 16 mẫu đầu tiên của x; với mỗi mẫu tính
    singular values của J (torch.linalg.svdvals); gom 16 mẫu bằng trung bình các đại lượng dưới
    đây (log10 trước, rồi trung bình), rồi báo cáo trung bình ± độ lệch chuẩn mẫu (ddof=1) qua 5 seed,
    3 chữ số có nghĩa (giống định dạng results.md):
      - log10 σ_max(J)
      - log10 RMS(σ) = log10 sqrt(mean_i σ_i²) = log10 (‖J‖_F / sqrt(64))
      - hạng số học: số σ_i > 64·eps64·σ_max (tương đương torch.linalg.matrix_rank mặc định), trên 64
      - log10 κ(J) = log10(σ_max/σ_min) CHỈ khi hạng số học = 64 VÀ σ_min > 64·eps64·σ_max
        (eps64 = 2.2e-16); nếu không, ghi "n/a" và nói lý do (hạng thiếu hoặc dưới ngưỡng số học;
        KHÔNG báo một condition number vô nghĩa do nhiễu làm tròn)
      - log10 của chặn trên bảo thủ c_J = prod_{k=2}^{L-1} ( ‖W^[k]‖_2 · max_i |σ'(z^[k-1]_i)| ) tại
        mẫu đó (dùng σ' thực đo được, không dùng 1/4), để so với σ_max.
    Với tỉ lệ "n/a" (ví dụ ReLU hạng thiếu) hãy báo thêm tỉ lệ (số cặp seed-mẫu) có hạng đủ.

B4. Tự kiểm tra bằng `assert` trong script (giống experiment.py):
    (a) σ_max(J) <= c_J·(1+1e-9) cho MỌI mẫu, seed, L, điều kiện.
    (b) Với một (seed, L, điều kiện) và một vector v ngẫu nhiên cố định: autograd cho
        ∂(vᵀ z^[L-1]) / ∂z^[1] trùng Jᵀv (allclose, rtol 1e-9), kiểm chứng đẳng thức
        δ^[1] = Jᵀ δ^[L-1] của mệnh đề ở C1.
    (c) Chạy cùng (seed,L,điều kiện) hai lần cho kết quả giống hệt (tính tất định).
    (d) Tái lập khởi tạo: in thêm c_ave = max σ'_toàn_cục · mean_k ‖W^[k]‖_2 (k=2..L, như
        experiment.py) và đối chiếu với cột "c (bound)" của results.md (vd. A L=5: 0.449±0.00761;
        B L=20: 2.68±0.0141); sai khác > 3 chữ số có nghĩa => dừng và báo lại.

B5. Đầu ra: results_jacobian.md cùng thư mục, dạng giống results.md (tiêu đề, dòng header gồm
    phiên bản torch/python, float32 init + float64 phổ, seeds, D, N, số mẫu = 16, bảng markdown).
    Chạy script HAI lần và `cmp` hai file kết quả: phải giống từng byte.
    Header phải ghi: "Jacobian trên chồng tầng ẩn 64x64 (z^[1] -> z^[L-1]); tầng ra 1 chiều bị loại".

---------------------------------------------------------------------
C. NỘI DUNG LATEX
---------------------------------------------------------------------
Số liệu mới trong văn bản và bảng PHẢI chép từ results_jacobian.md (mean ± std, đúng chữ số).
Mọi hình/bảng phải tuân thủ quy ước Đợt 1: dùng \captab (caption TRÊN bảng), nhãn tab:/fig:,
và có \ref đứng TRƯỚC phao. Thân bài giữ 3-5 hình (hiện 4); đợt này CHỈ thêm bảng, không thêm hình.
Nhận xét theo thứ tự: xu hướng chính -> cơ chế -> liên hệ giả thuyết; không đọc lại từng số.

C1. sections/03-nguyennhan.tex: thêm \subsection{Dynamical Isometry: phổ của toàn bộ Jacobian}
    (nhãn subsec:dyniso) ở CUỐI file (sau notebox kết thúc mục), gồm:
    (1) Định nghĩa J đúng như B2 (đúng thứ tự thừa số và ký hiệu D^[k] = diag(σ'(z^[k])), nhất quán
        với eq:jacforward và eq:deltarec). Nêu rõ J loại tầng ra vô hướng và vì sao (hạng 1).
    (2) MỆNH ĐỀ (infobox) + CHỨNG MINH đủ bước (phong cách như chứng minh ở 02-cosotoan.tex):
          δ^[1] = D^[1] (W^[2])ᵀ D^[2] (W^[3])ᵀ ... D^[L-2] (W^[L-1])ᵀ δ^[L-1] = Jᵀ δ^[L-1],
          σ_min(J)‖δ^[L-1]‖ <= ‖δ^[1]‖ <= σ_max(J)‖δ^[L-1]‖.
        Chứng minh: áp dụng eq:deltarec L-2 lần (quy nạp), nhận ra Jᵀ, rồi dùng bất đẳng thức
        SVD cho ma trận vuông 64x64. Nhấn mạnh: đây là chặn HAI chiều và đúng cho mọi vector
        δ^[L-1], trong khi c chỉ là chặn một chiều. Nêu hệ quả: σ_max(J) <= prod ‖W‖_2 ·
        max|σ'| = chặn kiểu c^{L-2} (đúng như ở eq:governing), còn σ_min và condition number
        κ(J) = σ_max/σ_min mới quyết định liệu MỌI hướng của tín hiệu lỗi có được truyền về hay
        không. Ghi chú: gradient theo W^[1] còn nhân thêm a^[0]ᵀ nên phép so sánh với chuẩn gradient
        chỉ có ý nghĩa ở bậc độ lớn (như đã nói ở 06).
    (3) Định nghĩa Dynamical Isometry (Xiao2018DynIso): phổ singular values của J tập trung quanh 1
        (mọi σ_i ≈ 1), nên ‖δ^[1]‖ ≈ ‖δ^[L-1]‖ cho mọi vector, không chỉ trung bình. So sánh với
        điều kiện "bảo toàn phương sai" (Xavier/He): nó chỉ kiểm soát một đại lượng trung bình bậc
        hai, tương ứng RMS(σ) trong B3. Chỉ giữ nhận xét này ở mức "gợi ý, dưới (G1)--(G3)" và chỉ giữ
        nếu số RMS(σ) đo được ủng hộ; nếu không, bỏ câu đó và nói rõ.
    (4) Một đoạn ngắn về hướng "khởi tạo trực giao" như một đường thứ tư đạt Dynamical Isometry mà
        không cần kết nối tắt [Xiao2018DynIso]; nói rõ tài liệu này KHÔNG thực nghiệm với khởi tạo
        trực giao. Chỉ phát biểu điều bạn đã kiểm chứng được từ tóm tắt bài báo (arXiv 1806.05393);
        không phát biểu thêm về ReLU hay CNN nếu chưa đọc.
    (5) Một câu dẫn về Yang et al. [Yang2019MeanField]: lý thuyết trường trung bình cho thấy BatchNorm
        ở mạng thuần (không kết nối tắt) có thể gây bùng nổ gradient ngay tại khởi tạo, tăng theo độ
        sâu; kết nối tắt làm giảm hiện tượng này. CHỈ giữ các vế bạn xác nhận được từ tóm tắt
        arXiv 1902.08129; nếu muốn viết "bất kể hàm kích hoạt" thì phải thấy chữ đó trong tóm tắt,
        nếu không thì bỏ vế đó. Nhãn thành phần dẫn đến số liệu ở C3.
    (6) Sửa nhẹ đoạn "Ba lối thoát này ... phân loại vét cạn" (khoảng dòng 335-353 của file, định vị
        bằng nội dung): bỏ khẳng định "vét cạn" vì khởi tạo trực giao / Dynamical Isometry là một
        hướng tiếp cận nữa; viết lại thành "ba lối thoát chính" và nhắc subsec:dyniso. Không đổi nội
        dung khoa học còn lại.

C2. sections/06-thucnghiem.tex: thêm \subsection{Phổ Jacobian đầu-cuối tại khởi tạo}
    (nhãn subsec:jacobian-exp) ngay SAU subsection "Phân tích" của thí nghiệm khởi tạo (trước
    guidelinebox "Cách đọc một bảng chuẩn gradient..."), gồm: một đoạn giao thức ngắn (mạng, 16 mẫu,
    float64 cho phổ, tầng ra loại bỏ, ngưỡng hạng/condition number), MỘT bảng
    (\captab, nhãn tab:jacobian-spectrum, caption TRÊN, độc lập: mô hình, L, số seed, số mẫu, câu
    phát hiện) với hàng A/B x L∈{5,10,20} và cột: log10 σ_max, log10 RMS(σ), hạng số học/64,
    log10 κ (hoặc n/a), log10 c_J; rồi phân tích: xu hướng -> cơ chế -> liên hệ với chặn c và
    với bảng grad-init. Chỉ rút ra điều số liệu cho thấy; nếu hạng của J (ReLU) thiếu, nêu rõ và
    giải thích bằng D^[k] có phần tử 0, không suy diễn thêm. Không in đậm ở bảng này (không có
    "tốt nhất" tự nhiên). Thêm \ref tới bảng trong đoạn giao thức, TRƯỚC bảng.

C3. sections/06-thucnghiem.tex: sau mục (c) của "Động lực học trong quá trình huấn luyện" và
    TRƯỚC notebox cuối, thêm \paragraph{(d) Khởi tạo chỉ đảm bảo ở $t=0$, và BatchNorm ở mạng
    thuần sâu không miễn nhiễm} với cấu trúc Cơ chế -> Bằng chứng -> Kết luận như (a)-(c):
    - Cơ chế (lý thuyết, nêu rõ là diễn giải không phải đo trực tiếp): He/Xavier chỉ cân bằng
      phương sai tại $t=0$ [theo (G1)--(G4) ở 03]; huấn luyện làm trôi phân phối trọng số nên giả thiết
      không còn đúng; BN ở mạng thuần sâu có thể gây bùng nổ gradient tại khởi tạo [Yang2019MeanField].
    - Bằng chứng: CHỈ dùng số đã có trong results_training.md (Lead đã đối chiếu; bạn phải tự kiểm
      lại từng số trước khi dùng). PlainNet ReLU+Kaiming+BN, log10ρ tại epoch 0: 0.862±0.0482
      (L=8), 1.56±0.0611 (L=16), 2.81±0.0832 (L=32), tức ρ>1 tại khởi tạo và tăng gần tuyến tính
      theo L (log10ρ), tương ứng stem gnorm 8.22, 36.8, 626; sau epoch 1: 0.0232±0.00475,
      −0.297±0.0768, −1.3±0.0419; sau epoch 10: 0.332±0.0716, −0.0106±0.144, −1.26±0.145.
      Đối chiếu: ResNet+BN giữ log10ρ trong [−0.914, 0.348] ở cả ba L (đã nêu ở (c)). Đối chiếu thêm
      PlainNet ReLU+Kaiming KHÔNG BN tại epoch 0: −0.0372, 0.0626, −0.133 (≈0, He hoạt động ở t=0).
      Sigmoid+Xavier+BN plain (log10ρ epoch 0 ≈ −0.16, −0.073, −0.052) KHÔNG cho thấy bùng nổ; nói
      thẳng điều đó như một quan sát và KHÔNG kết luận gì về lý thuyết Yang cho sigmoid.
      Dùng Bảng~\ref{tab:training-dynamics} (L=32) cho phần có sẵn; số của L=8,16 trích trực tiếp
      từ results_training.md trong câu văn (KHÔNG thêm bảng/hình mới trùng dữ liệu).
    - Kết luận có phạm vi: chỉ cho thiết lập này (MLP rộng 256, CIFAR-10 10k ảnh, SGD η=0.1, 3 hạt
      giống); KHÔNG tổng quát hoá cho BatchNorm nói chung. Không tuyên bố "BN gây bùng nổ" như một
      sự thật chung: nói "phù hợp với dự báo của [Yang2019MeanField]".
    - Không so sánh với khởi tạo trực giao (không có dữ liệu).

C4. sections/07-tongket.tex: viết lại đoạn "\textbf{BatchNorm và LayerNorm --- giữ $\mathbf{z}$ ngoài
    vùng bão hoà.}" (khoảng dòng 89-96) theo TA, gồm đủ các ý: (i) khởi tạo He/Xavier chỉ đảm bảo ở
    $t=0$; (ii) chuẩn hoá nhằm duy trì điều kiện đó trong huấn luyện nhưng KHÔNG có đảm bảo (bỏ
    cách nói "giúp duy trì" không điều kiện); (iii) BN ở mạng thuần rất sâu có thể gây bùng nổ
    gradient lúc khởi tạo [Yang2019MeanField], dẫn số liệu của C3; (iv) hướng khác: khởi tạo trực
    giao / Dynamical Isometry [Xiao2018DynIso], tài liệu không thực nghiệm; (v) trong các thí nghiệm
    ở đây, chỉ tổ hợp ResNet+BN giữ ρ gần 1 và để stem học được ở L=32 (không tuyên bố "rẻ nhất/
    bền nhất" vì không so với khởi tạo trực giao). Thêm \ref tới subsec:dyniso và
    tab:jacobian-spectrum nếu phù hợp. Đồng thời chỉnh câu mở đầu mục (dòng 10-14: "không phải một định
    lý phân loại") để nhất quán với C1(6). Giữ độ dài đoạn hợp lý.

C5. references.bib: KHÔNG thêm mục mới. Hai khoá Yang2019MeanField, Xiao2018DynIso phải được
    \cite ít nhất một lần trong sections/ (C1, C3, C4). Xiao có pages = {5393--5402}: Lead/User đã
    xác nhận giữ nguyên (nguồn Coder báo ở Đợt 1: PMLR v80).

---------------------------------------------------------------------
D. VIỆC MANG SANG TỪ ĐỢT 1 (bắt buộc, D5 phải đạt)
---------------------------------------------------------------------
D-a. In đậm ở Bảng training-dynamics (sections/06-thucnghiem.tex, các hàng "$0$ &"): BỎ in đậm ở
     epoch 0 (hai ô: PlainNet 10.6 ở khối BN và PlainNet 9.94 ở khối không BN; epoch 0 là mức khởi
     tạo ~10%). Chỉ in đậm giá trị Test acc trung bình tốt nhất giữa PlainNet và ResNet từ epoch 1
     đến epoch 10, và CHỈ khi cả hai cùng không phân kỳ (quy tắc cũ). Thêm vào caption một cụm ngắn
     giải thích: "in đậm: độ chính xác test trung bình cao hơn, từ epoch 1".
D-b. Triệt tiêu 2 cảnh báo Overfull trong sections/06-thucnghiem.tex (xác định vị trí bằng
     `grep -a -n 'Overfull' main.log`):
       (i) 26.47pt, đoạn "Phép đo" (g_stem = ∂L/∂W_stem ... và g_head = ∂L/∂W_head ... là công thức
           inline không ngắt dòng): đưa thành công thức hiển thị \[ ... \] hoặc dùng \allowbreak/
           ngắt câu; KHÔNG đổi nghĩa.
       (ii) 9.45pt, gạch đầu dòng "$\rho$ của ResNet ở gần $1$" (khoảng dòng 434-440): viết lại
           câu hoặc thêm \allowbreak; KHÔNG đổi số.
     Mọi Overfull mới do nội dung Đợt 2 (đặc biệt bảng mới và công thức dài) cũng phải xử lý.
D-c. Thông báo "LaTeX Warning: There were undefined references." (cuối main.log). Coder Đợt 1 nói là
     có sẵn ở baseline và PDF không có "??". Hãy CHẨN ĐOÁN căn nguyên bằng bằng chứng (vd.
     `grep -a -n -i 'undefined\|rerun' main.log`, đọc main.aux/main.bbl, thử bản sao ngoài repo)
     và hoặc triệt tiêu, hoặc giải thích có bằng chứng trong báo cáo vì sao vô hại. Lead sẽ chấp
     nhận "giải thích có bằng chứng" nếu không có Reference/Citation undefined thật.

---------------------------------------------------------------------
E. NGHIỆM THU (chạy đúng lệnh, báo pass/fail kèm lệnh và output rút gọn)
---------------------------------------------------------------------
E1. Citations resolve:
      grep -n 'cite.*Yang2019MeanField' sections/*.tex ; grep -n 'cite.*Xiao2018DynIso' sections/*.tex
      -> mỗi khoá >= 1 kết quả. Sau build: `pdftotext main.pdf - | grep -ni 'mean field theory of
      batch normalization\|Dynamical isometry'` thấy cả hai mục trong danh sách tài liệu tham khảo;
      `grep -a -c "Citation \`" main.log` = 0; `pdftotext main.pdf - | grep -c '\[?\]\|??'` = 0.
E2. Quy ước phao của Đợt 1 vẫn đạt: chạy lại script D1 (thứ tự \ref trước \label, glob
      'sections/0[1-8]-*.tex') -> toàn PASS, gồm nhãn mới tab:jacobian-spectrum;
      `grep -nE '(Bảng|Hình)[ ~]*[0-9]' main.tex sections/*.tex` rỗng; caption bảng trước tabular.
E3. Số liệu khớp: mọi con số của Bảng tab:jacobian-spectrum và mọi số mới trong C2/C3/C4 khớp
      results_jacobian.md hoặc results_training.md từng chữ số (trình bày bảng đối chiếu: ô nào
      của LaTeX <- dòng nào của file nguồn). Chạy `python3 experiment_jacobian.py` hai lần và
      `cmp` kết quả: giống từng byte. Mọi assert ở B4 pass (in "OK").
E4. Toán: Lead sẽ đọc mệnh đề ở C1(2). Trong báo cáo, dán lại phát biểu mệnh đề, các bước chứng
      minh, và xác nhận thứ tự thừa số/chuyển vị khớp eq:jacforward, eq:deltarec và script
      (B4b đã kiểm đẳng thức δ^[1]=Jᵀδ^[L-1] bằng autograd).
E5. Build sạch: `bash build.sh` (3 lượt pdflatex + bibtex, tuần tự).
      `grep -a -n 'Overfull' main.log` -> không dòng nào > 1pt (mục tiêu: 0 dòng);
      0 dòng bắt đầu bằng "!" ; không "Reference ... undefined" / "Citation ... undefined".
      Cảnh báo có sẵn được phép: biblatex csquotes/Language 'vietnamese', pgfplots compatibility,
      LaTeX Font Warning cỡ 5.5; riêng D-c xem mục D.
E6. Không đụng dữ liệu gốc: `git status --short` và `git diff --stat -- results_training.json
      results_training.md results.md experiment.py experiment_training_dynamics.py
      experiment_numpy_legacy.py figures` -> RỖNG. File mới được phép xuất hiện: experiment_jacobian.py,
      results_jacobian.md, TASK_REPORT_DOT1.md, sections/08-phuluc.tex (đã có), TASK_*.md.
E7. Bố cục: số trang trước/sau (hiện 33); thân bài vẫn 4 hình, Hình 5 ở Phụ lục; bảng đánh số
      tự động liền mạch (bảng Jacobian mới sẽ chen vào, các bảng sau tự dịch số); đọc ảnh các trang
      chứa bảng mới và mệnh đề, xác nhận không tràn lề.
E8. Phạm vi: `grep -n -i 'adam\|R\^2\|decay_slope\|Hạn chế' sections/*.tex` không có nội dung mới
      thuộc Đợt 3.

---------------------------------------------------------------------
F. BÁO CÁO CUỐI -> GHI VÀO TASK_REPORT.md (sau khi `cp TASK_REPORT.md TASK_REPORT_DOT1.md`)
---------------------------------------------------------------------
  1. Danh sách file đã sửa/tạo (1 dòng mỗi file).
  2. Bảng nghiệm thu E1..E8: Pass/Fail, lệnh đã chạy, output rút gọn.
  3. Bảng đối chiếu số liệu (E3) và kết quả các assert B4.
  4. Phát biểu mệnh đề C1(2) + các bước chứng minh (E4).
  5. Với mỗi phát biểu về Yang2019 và Xiao2018 trong văn bản: câu nguyên văn trong tóm tắt/nguồn
     mà bạn dùng để kiểm chứng (hoặc ghi rõ "bỏ vì không kiểm chứng được").
  6. Kết quả D-a, D-b, D-c (căn nguyên "undefined references").
  7. Số trang trước/sau; mọi điểm KHÔNG làm được / tự quyết khác prompt / cần Lead-User quyết.
  8. Không commit. Dừng sau khi ghi TASK_REPORT.md; chờ Lead nghiệm thu trước khi sang Đợt 3.
```
