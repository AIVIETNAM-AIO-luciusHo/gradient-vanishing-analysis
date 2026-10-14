# End-to-end Jacobian spectrum at initialisation (mean +/- std, ddof=1, over 5 seeds)

`torch 2.14.0+cpu | python 3.12.3 | init float32 CPU -> spectrum float64 | seeds=[42, 100, 2024, 2026, 9999] | D=64 N=256 | samples per seed = 16 (first 16 rows of x)`

Jacobian trên chồng tầng ẩn 64x64 (z^[1] -> z^[L-1]); tầng ra 1 chiều bị loại.
J = W^[L-1] D^[L-2] ... W^[2] D^[1], L-2 thừa số. Mỗi seed: trung bình của log10 trên 16 mẫu; bảng: trung bình ± độ lệch chuẩn qua 5 seed.
Hạng số học = số sigma_i > 64*eps64*sigma_max. log10 kappa chỉ tính khi hạng = 64 (mọi sigma_i trên ngưỡng); nếu không: n/a.

| condition | L | log10 sigma_max(J) | log10 RMS(sigma) | numerical rank (of 64) | full-rank pairs (of 80) | log10 kappa(J) | log10 c_J | c_ave (reproduces results.md) |
|---|---|---|---|---|---|---|---|---|
| A: sigmoid + Xavier | 5 | -1.45 ± 0.0329 | -1.93 ± 0.0165 | 64 ± 0 | 80/80 | 5.72 ± 0.811 | -0.937 ± 0.0238 | 0.449 ± 0.00761 |
| A: sigmoid + Xavier | 10 | -4.39 ± 0.0528 | -5.05 ± 0.024 | 64 ± 0 | 80/80 | 11.6 ± 0.781 | -2.52 ± 0.0389 | 0.469 ± 0.00277 |
| A: sigmoid + Xavier | 20 | -10.6 ± 0.0672 | -11.3 ± 0.0475 | 61 ± 0 | 0/80 | n/a (rank-deficient or sigma_min < 64*eps64*sigma_max) | -5.63 ± 0.0249 | 0.479 ± 0.00293 |
| B: ReLU + He | 5 | 0.557 ± 0.0219 | -0.00502 ± 0.0252 | 28.9 ± 0.872 | 0/80 | n/a (rank-deficient or sigma_min < 64*eps64*sigma_max) | 1.32 ± 0.0237 | 2.4 ± 0.0379 |
| B: ReLU + He | 10 | 0.688 ± 0.0872 | -0.0286 ± 0.102 | 26.6 ± 2.56 | 0/80 | n/a (rank-deficient or sigma_min < 64*eps64*sigma_max) | 3.5 ± 0.039 | 2.59 ± 0.0179 |
| B: ReLU + He | 20 | 0.676 ± 0.129 | -0.122 ± 0.124 | 23.6 ± 1.09 | 0/80 | n/a (rank-deficient or sigma_min < 64*eps64*sigma_max) | 7.92 ± 0.0249 | 2.68 ± 0.0141 |
