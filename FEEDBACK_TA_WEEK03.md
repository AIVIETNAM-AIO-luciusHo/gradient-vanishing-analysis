# [AIO 2026] FEEDBACK & GUIDELINES TỪ TA - CHUYÊN ĐỀ GRADIENT VANISHING
**Nguồn trích xuất:** `Answer Week03.pdf` — AI VIET NAM (EDA2026)[cite: 25, 26]  
**Người giải đáp:** Trần Chí Công, Võ Thị Thanh Kiều[cite: 25]  
**Mục tiêu áp dụng:** Dùng làm kim chỉ nam (Rubrics & Directives) để `module4-leader` rà soát, phản biện và nâng cấp tài liệu `main.tex` / `main.pdf` (32 trang).

---

## PHẦN 1: CHUYÊN ĐỀ GRADIENT VANISHING (TRANG 6 - 8)

### 1. Quy tắc trình bày Figure/Table và nhận xét kết quả trong Báo cáo
- **Cách dẫn và vị trí:**
  - Mọi hình/bảng đưa vào bài bắt buộc phải được nhắc và dẫn dắt trong nội dung văn bản trước khi xuất hiện (ví dụ: *"Hình 3 cho thấy..."*). Tuyệt đối không để hình/bảng nằm mồ côi mà không có phân tích[cite: 30].
  - Quy ước vị trí: **Caption của bảng đặt phía trên; caption của hình đặt phía dưới**[cite: 30].
- **Tiêu chuẩn của Caption:**
  - Caption phải mang tính độc lập (tự đọc hiểu được mà không cần đọc toàn bài): nêu rõ vẽ cái gì, thiết lập thử nghiệm nào (mô hình, độ sâu mạng, số lượng seed) và kèm 1 câu tóm tắt phát hiện chính[cite: 30].
- **Thứ tự nhận xét khoa học:**
  $$\text{Mô tả xu hướng chính} \longrightarrow \text{Giải thích cơ chế bản chất} \longrightarrow \text{Liên hệ với giả thuyết / câu hỏi nghiên cứu}$$
  Không đọc lại từng con số trong bảng; chỉ nhấn mạnh các con số có ý nghĩa bước ngoặt[cite: 30].
- **Quy chuẩn đồ họa biểu đồ gradient:**
  - Trục $y$ bắt buộc để thang đo **log scale**[cite: 30].
  - Ghi rõ phương pháp chuẩn hóa[cite: 30].
  - Giữ cùng một bảng màu cố định cho cùng một phương pháp trên mọi biểu đồ[cite: 30].
  - Kết quả báo cáo phải ở dạng **$\text{mean} \pm \text{standard deviation}$** qua các seed; in đậm giá trị tốt nhất trong bảng[cite: 30].
  - Thân bài chỉ giữ 3–5 hình then chốt; các biểu đồ phụ chuyển về Phụ lục[cite: 31]. Tránh đưa cùng một dữ liệu lên cả bảng lẫn hình[cite: 31].

---

### 2. Chỉ số đánh giá: `decay_slope` từng layer vs. Tổng suy giảm toàn mạng
- **Vấn đề thực nghiệm:** `decay_slope` giảm khi độ sâu tăng (từ 1.12 ở 3 layer xuống 0.47 ở 25 layer), nhưng tổng suy giảm thực tế lại tăng do số lượng tầng nhân dồn[cite: 31].
- **Chỉ đạo kỹ thuật từ TA:**
  - **Chỉ số chính (Primary Metric):** Lấy **Tổng suy giảm toàn mạng** làm chỉ số trung tâm:
    $$\text{Total Degradation} = \log\left( \frac{\Vert{}\nabla_{W_{\text{first}}} \mathcal{L}\Vert{}}{\Vert{}\nabla_{W_{\text{last}}} \mathcal{L}\Vert{}} \right)$$
    *Lý do:* Câu hỏi cốt lõi của nghiên cứu là lượng gradient truyền ngược về được layer đầu tiên còn lại bao nhiêu, và đó là yếu tố quyết định tầng đầu có cập nhật được hay không[cite: 31].
  - **Chỉ số phụ (Secondary Metric):** Dùng `decay_slope` mỗi layer để giải thích tốc độ hao hụt cục bộ[cite: 31]. Cần báo cáo kèm hệ số xác định $R^2$ để chứng minh mô hình phân rã theo cấp số nhân ($e^{-\alpha l}$) có còn khớp chuẩn xác ở các mạng sâu hay không[cite: 31].

---

### 3. Cơ chế giải thích hiện tượng Adam sụp đổ ở mạng sâu
- **Vấn đề thực nghiệm:** Adam hoạt động rất tốt đến 7 layer (Accuracy $0.858$) nhưng sụp đổ hoàn toàn ở 15 layer (Accuracy $0.10$, $\text{drift\_L1} = 0$)[cite: 31].
- **Bản chất cơ chế giải tích:**
  Công thức cập nhật tham số của thuật toán Adam:
  $$\Delta \theta = -\eta \frac{\hat{m}}{\sqrt{\hat{v}} + \epsilon}, \quad \text{với } \epsilon = 10^{-8} \text{ (mặc định)}$$
  - Khi gradient tại các tầng đầu ở mạng sâu (15 layers) suy giảm cực nhỏ (chạm ngưỡng $\le \epsilon$), bước cập nhật thực tế bị suy biến về xấp xỉ 0, dẫn tới $\text{drift\_L1} = 0$[cite: 31].
- **Yêu cầu hoàn thiện trong bài:**
  - Ghi log độ lớn gradient của layer đầu đối chiếu với giá trị $\epsilon$[cite: 31].
  - Thử nghiệm kiểm chứng với $\epsilon$ nhỏ hơn (ví dụ $\epsilon = 10^{-12}$) để chứng minh giới hạn bù cập nhật của Adam[cite: 31].
  - Kiểm tra loại trừ lỗi kỹ thuật ($\text{NaN}$ hoặc lỗi hàm log)[cite: 31].
  - Giới hạn phạm vi kết luận trong thiết lập mạng đang xét, không khái quát hóa tiêu cực cho Adam nói chung[cite: 31].

---

### 4. Tính ổn định phương sai: He/Kaiming + BatchNorm vs. Residual Connection
- **Vấn đề lý thuyết:** Kết hợp He/Kaiming với BatchNorm/LayerNorm có duy trì ổn định phương sai gradient ở các tầng đầu qua nhiều epoch không, hay bắt buộc phải có Residual Connection[cite: 31]?
- **Giải đáp từ TA:**
  - **Không có gì đảm bảo duy trì ổn định:** Khởi tạo He chỉ bảo toàn phương sai tại **thời điểm $t = 0$ (lúc khởi tạo)**; trong quá trình huấn luyện, phân phối trọng số bị trôi lệch (drift) và giả định ban đầu bị phá vỡ hoàn toàn[cite: 31].
  - **Mặt trái của BatchNorm ở mạng Plain sâu:** Dù BatchNorm ổn định chiều lan truyền tiến (forward pass), nhưng ở các mạng plain rất sâu, chính BatchNorm lại có thể kích hoạt hiện tượng bùng nổ gradient ngay lúc khởi tạo[cite: 31].
    > *Tài liệu tham khảo bắt buộc trích dẫn:* **Yang et al., 2019**, *"A Mean Field Theory of Batch Normalization"*[cite: 31].
  - Bản thân bài báo gốc của ResNet (He et al., 2016) cũng đã chỉ ra mạng plain có BatchNorm vẫn gặp hiện tượng suy giảm hiệu năng (degradation) khi tăng độ sâu[cite: 31].
  - **Giải pháp thay thế:** Không chỉ có Residual; phương pháp **Orthogonal Initialization** nhằm đạt trạng thái *Dynamical Isometry* (**Xiao et al., 2018**) vẫn huấn luyện thành công mạng rất sâu mà không cần skip connection[cite: 32]. Tuy nhiên, Residual Connection vẫn là giải pháp thực nghiệm rẻ và bền vững nhất[cite: 32].
  - *Minh chứng thực nghiệm:* Tận dụng dữ liệu CIFAR-10 đã chạy để vẽ đường gradient norm tầng đầu qua các epoch nhằm trả lời trực tiếp câu hỏi này[cite: 32].

---

### 5. Phân tích Phổ trị riêng & Giá trị kỳ dị (Dynamical Isometry)
- **Vấn đề:** Phân tích phổ trị riêng của từng ma trận trọng số $W_l$ đã đúng góc nhìn Dynamical Isometry chưa[cite: 32]?
- **Chỉ đạo kỹ thuật từ TA:**
  - **Chưa đủ:** Dynamical Isometry định nghĩa trên phổ giá trị kỳ dị của **toàn bộ ma trận Jacobian ánh xạ từ đầu vào đến đầu ra**[cite: 32]:
    $$J = \prod_{l=1}^L D_l W_l$$
    trong đó $D_l = \text{diag}(\sigma'(z_l))$ là ma trận đường chéo chứa đạo hàm của hàm kích hoạt[cite: 32].
  - Phổ của riêng $W_l$ mới chỉ phản ánh một nửa vấn đề, đúng như công thức chặn trên:
    $$c = \max \sigma' \cdot \sigma_{\max}(W)$$
  - **Đề xuất nâng cấp:** 
    - Tính toán hoặc ước lượng phổ Jacobian end-to-end $J$ trên một số batch đại diện (sử dụng `torch.autograd.functional.jacobian` cho mạng nhỏ hoặc thuật toán *Power Iteration* để lấy các giá trị kỳ dị lớn nhất)[cite: 32].
    - Bổ sung **Condition Number của Jacobian**[cite: 32].
    - Bỏ qua *Gradient Norm Variance* phức tạp; thay bằng **Tỉ số Gradient Norm giữa tầng đầu và tầng cuối theo epoch** để trực quan và mạch lạc hơn[cite: 32].

---

### 6. Rà soát Outline & Thiết lập Thí nghiệm
- **Ghi rõ siêu tham số nền:** Xác định rõ hàm kích hoạt và khởi tạo cụ thể trong Stage A (ví dụ: Sigmoid + Xavier) thay vì ghi chung chung là "cố định"[cite: 32].
- **Định nghĩa chuẩn toán học:**
  - Công thức rõ ràng cho RMS Gradient và Normalized Weight Change[cite: 32].
  - Quy định thời điểm lấy mẫu (vài trăm iteration đầu hay cuối mỗi epoch) và lấy trung bình trên bao nhiêu batch[cite: 32].
- **Kiểm định giả thuyết H2:** Khi dùng thuật ngữ tương quan ("associated"), phải chỉ rõ phương pháp kiểm định (ví dụ: tương quan Pearson/Spearman giữa gradient tầng đầu và weight change, hoặc số epoch cần để loss hội tụ dưới một ngưỡng)[cite: 32].
- **Lưu ý Learning Rate:** Dùng chung một learning rate cho mọi độ sâu có thể khiến mạng sâu thất bại vì bước học không phù hợp chứ không hẳn do gradient vanishing $\implies$ Ghi rõ hạn chế này hoặc quét nhanh 2–3 giá trị learning rate ở mạng sâu nhất[cite: 32].

---

## PHẦN 2: CÁC CÂU HỎI BỔ TRỢ LIÊN QUAN (MỤC I - DEEP LEARNING, TRANG 4)

### 1. Đánh giá Activation Function & Weight Initialization (Trang 4)
- Khi đánh giá ma trận phối hợp giữa Hàm kích hoạt và Khởi tạo trọng số, giữ cố định learning rate chưa hoàn toàn công bằng vì learning rate tối ưu của mỗi cặp là khác nhau (Sigmoid cần learning rate lớn, trong khi ReLU + He với learning rate lớn dễ phân kỳ)[cite: 28].
- Nếu không đủ thời gian sweep learning rate, cần đưa luận điểm này vào phần **Hạn chế của nghiên cứu (Limitations)**[cite: 28].

### 2. Bổ sung chỉ số đánh giá khả năng lan truyền tín hiệu (Trang 4)
Bên cạnh Gradient Norm và Activation Variance, 2 chỉ số có giá trị học thuật cao cần bổ sung[cite: 28]:
1. **Tỉ lệ nơ-ron bão hòa / nơ-ron chết:** Đo tỉ lệ phần trăm các unit bị bão hòa (vùng đạo hàm triệt tiêu của Sigmoid/Tanh) hoặc bị chết (vùng âm triệt tiêu của Dead ReLU) theo từng layer[cite: 28].
2. **Tỉ lệ cập nhật trọng số:**
   $$\text{Relative Update Ratio} = \frac{\Vert{}\Delta W\Vert{}_F}{\Vert{}W\Vert{}_F}$$
   Chỉ số này chứng minh trực tiếp liệu gradient nhỏ có thực sự làm các tầng nông ngừng học hay không[cite: 28].

---

## PHẦN 3: CHECKLIST DÀNH CHO AGENT `module4-leader` RÀ SOÁT `main.tex`

Mỗi khi rà soát file `main.tex`, `module4-leader` phải đối chiếu với 6 tiêu chí nghiệm thu sau:

- [ ] **Tiêu chí 1 (Biểu đồ):** Mọi biểu đồ gradient đã dùng thang đo `log scale` trên trục $y$, có ghi chú rõ ràng về seed, độ sâu mạng và nhận xét trước khi xuất hiện trong bài chưa[cite: 30]?
- [ ] **Tiêu chí 2 (Chỉ số suy giảm):** Bài viết đã đưa ra công thức tính **Tổng suy giảm toàn mạng (Log ratio layer đầu/cuối)** làm chỉ số chính bên cạnh `decay_slope` chưa[cite: 31]?
- [ ] **Tiêu chí 3 (Cơ chế Adam):** Phần phân tích Adam sụp đổ ở mạng sâu đã trích dẫn công thức cập nhật kèm hằng số $\epsilon = 10^{-8}$ để giải thích cơ chế chưa[cite: 31]?
- [ ] **Tiêu chí 4 (Lý thuyết BatchNorm):** Đã trích dẫn bài báo lý thuyết Mean Field Theory của **Yang et al. (2019)** và giải thích hiện tượng He init bị drift trong quá trình train chưa[cite: 31]?
- [ ] **Tiêu chí 5 (Dynamical Isometry):** Đã mở rộng công thức phổ trị riêng sang toàn bộ ma trận Jacobian end-to-end $J = \prod_l D_l W_l$ (bao gồm đạo hàm hàm kích hoạt $D_l$) và trích dẫn **Xiao et al. (2018)** chưa[cite: 32]?
- [ ] **Tiêu chí 6 (Tính minh bạch):** Đã nêu rõ các hạn chế về cố định learning rate và phân biệt rõ hiện tượng quan sát được với cơ chế giải tích chưa[cite: 28, 31, 32]?

