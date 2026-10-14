# TASK_DISPATCH — ĐỢT COMMIT (do module4-leader ghi; User đã duyệt)

> **Coder:** đọc toàn bộ file này, làm ĐÚNG các bước dưới đây rồi ghi báo cáo vào `TASK_REPORT_DOT3.md`
> (KHÔNG ghi đè `TASK_REPORT.md`). Lead KHÔNG commit; chỉ Coder commit. Bản Đợt 3 lưu ở
> `TASK_DISPATCH_DOT3.md`. KHÔNG sửa file này, KHÔNG sửa nội dung bất kỳ file nào khác.

Quyết định của User: commit CÓ; hình đường ρ có BN: BỎ; bảng màu cố định cho 2 PNG: BỎ.
Rủi ro đã chấp nhận: abstract Yang/Xiao chưa kiểm độc lập; căn nguyên "undefined references" chưa xác minh.

```text
VAI TRÒ: Coder của dự án "Gradient Vanishing" (Module 04). Thư mục:
  /home/lucius/Project/Learning-AI-engineer/docs/Module-04/gradient-vanishing/
Chỉ làm đúng các bước 1-6. Không sửa file nào. Không build lại. KHÔNG push, KHÔNG amend, KHÔNG rebase,
KHÔNG `git add -A`, KHÔNG `git add .`.

BƯỚC 1. Chạy `git status --short` và ghi nguyên văn vào báo cáo. Xác nhận main.pdf mới hơn mọi file .tex:
  `find main.tex sections -name '*.tex' -newer main.pdf` phải RỖNG. Nếu không rỗng: DỪNG, báo lại, không commit.

BƯỚC 2. `git add` CHÍNH XÁC các file sau (liệt kê tường minh trên dòng lệnh):
  main.tex main.pdf references.bib sections/*.tex
  experiment_jacobian.py analysis_decay.py results_jacobian.md results_decay.md
  FEEDBACK_TA_WEEK03.md
  TASK_DISPATCH.md TASK_DISPATCH_DOT1.md TASK_DISPATCH_DOT2.md TASK_DISPATCH_DOT3.md
  TASK_REPORT.md TASK_REPORT_DOT1.md TASK_REPORT_DOT2.md
  KHÔNG add: experiment_stdout.txt, training_dynamics_sanity.log, TASK_REPORT_DOT3.md (tạo sau commit).

BƯỚC 3. Chạy `git status --short` và `git diff --cached --stat`. Danh sách staged phải KHỚP đúng danh sách
  bước 2 (sections/*.tex gồm 01-08; các file chưa đổi như 01, 05 không xuất hiện trong diff là bình thường).
  DỪNG và báo lại, KHÔNG commit, nếu:
  (i) có file lạ staged, đặc biệt results_training.*, results.md, experiment.py, experiment_training_dynamics.py,
      experiment_numpy_legacy.py, figures/*, build.sh;
  (ii) có file trong danh sách bị .gitignore loại (dùng `git check-ignore -v <file>` để kiểm khi thiếu);
  (iii) experiment_stdout.txt hoặc training_dynamics_sanity.log xuất hiện trong staged.

BƯỚC 4. Commit với đúng message sau (dùng `git commit -F` từ file tạm trong scratchpad/`/tmp`, không để file
  tạm trong repo):

docs(gradient-vanishing): TA week-3 revision (33 -> 40 pages)

- captions/labels: \captab counter, above-table/below-figure, fig:gradnorm moved to appendix A
- 03: Dynamical Isometry section with two-sided Jacobian proposition + proof
- 06: end-to-end Jacobian spectrum, Total Degradation + decay_slope/R^2, BN-at-init, limitations section, theoretical Adam note
- 07: BatchNorm/He paragraph rewritten, claims softened
- refs: Yang2019MeanField, Xiao2018DynIso
- results_jacobian.md, results_decay.md derived from existing data only

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>

BƯỚC 5. KHÔNG push. KHÔNG amend/rebase.

BƯỚC 6. Ghi `TASK_REPORT_DOT3.md` (file mới, KHÔNG add, KHÔNG commit lại) gồm: (a) `git status --short` bước 1,
  (b) kết quả `find ... -newer`, (c) `git diff --cached --stat` bước 3, (d) hash commit và
  `git log --oneline -5`, (e) `git status --short` sau commit (phải chỉ còn experiment_stdout.txt,
  training_dynamics_sanity.log, TASK_REPORT_DOT3.md là untracked), (f) mọi điểm lệch/không làm được.
  Dừng sau báo cáo.
```
