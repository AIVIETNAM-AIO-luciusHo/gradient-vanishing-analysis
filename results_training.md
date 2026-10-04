# Phase 2: training dynamics, plain vs resnet

- torch 2.14.0+cpu | python 3.12.3 | float32 CPU, 6 threads, deterministic algorithms on
- dataset: CIFAR-10 (train subset 10000 fixed by seed 0, full 10k test) at /home/lucius/.cache/cifar10
- SGD lr=0.1 momentum=0 wd=0 | batch 128 | 10 epochs | probe batch 512 | width 256
- seeds [42, 100, 2024] | depths [8, 16, 32] | eps32 = 1.19e-07
- tables: mean ± std (ddof=1) over seeds, 3 sig. figs; (n=k) = only k seeds finite/not diverged
- total runtime 17.4 min (grid 17.3 min)

## log10(rho) = log10(||g_stem|| / ||g_head||)

| arch | L | act/init | BN | epoch 0 | epoch 1 | epoch 5 | epoch 10 |
|---|---|---|---|---|---|---|---|
| plain | 8 | relu+kaiming | off | -0.0372 ± 0.0977 | 0.0948 ± 0.172 | 0.222 ± 0.121 | 0.328 ± 0.0488 |
| resnet | 8 | relu+kaiming | off | -0.279 ± 0.174 | n/a | n/a | n/a |
| plain | 16 | relu+kaiming | off | 0.0626 ± 0.139 | 0.121 ± 0.0846 | 0.342 ± 0.141 | 0.305 ± 0.0909 |
| resnet | 16 | relu+kaiming | off | -0.491 ± 0.157 | n/a | n/a | n/a |
| plain | 32 | relu+kaiming | off | -0.133 ± 0.191 | 0.221 ± 0.317 (n=2) | 0.384 ± 0.0145 (n=2) | 0.0497 ± 0.545 (n=2) |
| resnet | 32 | relu+kaiming | off | -0.737 ± 0.0462 | n/a | n/a | n/a |
| plain | 8 | relu+kaiming | on | 0.862 ± 0.0482 | 0.0232 ± 0.00475 | 0.187 ± 0.129 | 0.332 ± 0.0716 |
| resnet | 8 | relu+kaiming | on | 0.2 ± 0.0665 | -0.674 ± 0.0702 | -0.415 ± 0.0941 | -0.122 ± 0.0954 |
| plain | 16 | relu+kaiming | on | 1.56 ± 0.0611 | -0.297 ± 0.0768 | -0.247 ± 0.0737 | -0.0106 ± 0.144 |
| resnet | 16 | relu+kaiming | on | 0.3 ± 0.0452 | -0.731 ± 0.184 | -0.488 ± 0.0698 | -0.25 ± 0.0925 |
| plain | 32 | relu+kaiming | on | 2.81 ± 0.0832 | -1.3 ± 0.0419 | -1.36 ± 0.154 | -1.26 ± 0.145 |
| resnet | 32 | relu+kaiming | on | 0.348 ± 0.101 | -0.914 ± 0.0523 | -0.569 ± 0.145 | -0.342 ± 0.156 |
| plain | 8 | sigmoid+xavier | off | -5.05 ± 0.0904 | -5.16 ± 0.114 | -5.02 ± 0.0995 | -4.82 ± 0.105 |
| resnet | 8 | sigmoid+xavier | off | -2.83 ± 0.106 | -2.99 ± 0.151 | -2.8 ± 0.0838 | -2.57 ± 0.126 |
| plain | 16 | sigmoid+xavier | off | -10.1 ± 0.126 | -10.2 ± 0.181 | -10 ± 0.0824 | -9.84 ± 0.146 |
| resnet | 16 | sigmoid+xavier | off | -5.4 ± 0.11 | -5.55 ± 0.161 | -5.37 ± 0.0814 | -5.13 ± 0.147 |
| plain | 32 | sigmoid+xavier | off | -20.5 ± 0.101 | -20.4 ± 0.131 | -20.3 ± 0.161 | -20.1 ± 0.132 |
| resnet | 32 | sigmoid+xavier | off | -10.8 ± 0.103 | -10.8 ± 0.128 | -10.6 ± 0.101 | -10.3 ± 0.0984 |
| plain | 8 | sigmoid+xavier | on | -0.16 ± 0.0756 | -0.196 ± 0.173 | -0.00282 ± 0.127 | 0.0941 ± 0.122 |
| resnet | 8 | sigmoid+xavier | on | -0.347 ± 0.0787 | -0.443 ± 0.2 | -0.215 ± 0.137 | -0.0564 ± 0.149 |
| plain | 16 | sigmoid+xavier | on | -0.0732 ± 0.119 | -0.165 ± 0.184 | -0.00444 ± 0.133 | 0.103 ± 0.143 |
| resnet | 16 | sigmoid+xavier | on | -0.265 ± 0.109 | -0.425 ± 0.213 | -0.203 ± 0.148 | -0.0271 ± 0.104 |
| plain | 32 | sigmoid+xavier | on | -0.0524 ± 0.0885 | -0.142 ± 0.147 | -0.0711 ± 0.13 | 0.0398 ± 0.157 |
| resnet | 32 | sigmoid+xavier | on | -0.242 ± 0.078 | -0.409 ± 0.196 | -0.227 ± 0.142 | -0.0447 ± 0.131 |

## stem ||g||_F

| arch | L | act/init | BN | epoch 0 | epoch 1 | epoch 5 | epoch 10 |
|---|---|---|---|---|---|---|---|
| plain | 8 | relu+kaiming | off | 3.38 ± 0.622 | 2.32 ± 0.249 | 1.93 ± 0.485 | 3.28 ± 0.219 |
| resnet | 8 | relu+kaiming | off | 24.2 ± 7.48 | n/a | n/a | n/a |
| plain | 16 | relu+kaiming | off | 2.51 ± 0.597 | 1.62 ± 1.08 | 1.37 ± 0.567 | 2.14 ± 0.182 |
| resnet | 16 | relu+kaiming | off | 96.8 ± 28 | n/a | n/a | n/a |
| plain | 32 | relu+kaiming | off | 2.95 ± 0.9 | 0.589 ± 0.423 (n=2) | 0.891 ± 0.503 (n=2) | 5.41 ± 3.08 (n=2) |
| resnet | 32 | relu+kaiming | off | 2.32e+03 ± 244 | n/a | n/a | n/a |
| plain | 8 | relu+kaiming | on | 8.22 ± 0.32 | 1.23 ± 0.0171 | 1.25 ± 0.0535 | 1.57 ± 0.0678 |
| resnet | 8 | relu+kaiming | on | 10.4 ± 0.261 | 1.75 ± 0.075 | 1.97 ± 0.0757 | 2.39 ± 0.206 |
| plain | 16 | relu+kaiming | on | 36.8 ± 1.07 | 0.569 ± 0.0671 | 0.528 ± 0.043 | 0.789 ± 0.046 |
| resnet | 16 | relu+kaiming | on | 20.5 ± 0.997 | 1.19 ± 0.0841 | 1.59 ± 0.0845 | 2.27 ± 0.676 |
| plain | 32 | relu+kaiming | on | 626 ± 59.4 | 0.0735 ± 0.00931 | 0.0531 ± 0.00477 | 0.0469 ± 0.0076 |
| resnet | 32 | relu+kaiming | on | 45 ± 1.39 | 0.701 ± 0.0672 | 0.993 ± 0.401 | 1.07 ± 0.221 |
| plain | 8 | sigmoid+xavier | off | 1.24e-05 ± 1.2e-06 | 1.26e-05 ± 1.19e-06 | 1.19e-05 ± 2.1e-07 | 1.15e-05 ± 7.51e-07 |
| resnet | 8 | sigmoid+xavier | off | 0.00335 ± 0.000162 | 0.00363 ± 0.000253 | 0.00344 ± 0.00022 | 0.00337 ± 6.52e-05 |
| plain | 16 | sigmoid+xavier | off | 1.16e-10 ± 1.34e-11 | 1.17e-10 ± 1e-11 | 1.11e-10 ± 8.33e-12 | 1.08e-10 ± 7.08e-12 |
| resnet | 16 | sigmoid+xavier | off | 8.77e-06 ± 9.84e-07 | 9.63e-06 ± 4.53e-07 | 9.18e-06 ± 6.7e-07 | 9.32e-06 ± 2.98e-07 |
| plain | 32 | sigmoid+xavier | off | 7.05e-21 ± 5.51e-22 | 6.84e-21 ± 7.53e-22 | 6.27e-21 ± 1.04e-21 | 5.71e-21 ± 1.56e-21 |
| resnet | 32 | sigmoid+xavier | off | 5.99e-11 ± 5.91e-12 | 6.13e-11 ± 9.31e-12 | 5.84e-11 ± 4.35e-12 | 5.78e-11 ± 5.09e-12 |
| plain | 8 | sigmoid+xavier | on | 1.13 ± 0.0138 | 1.04 ± 0.223 | 1.32 ± 0.152 | 1.56 ± 0.0738 |
| resnet | 8 | sigmoid+xavier | on | 1.1 ± 0.0349 | 1.01 ± 0.121 | 1.48 ± 0.0936 | 1.66 ± 0.0449 |
| plain | 16 | sigmoid+xavier | on | 1.18 ± 0.0275 | 1.12 ± 0.171 | 1.39 ± 0.248 | 1.4 ± 0.132 |
| resnet | 16 | sigmoid+xavier | on | 1.16 ± 0.0547 | 1.08 ± 0.122 | 1.42 ± 0.0855 | 1.67 ± 0.251 |
| plain | 32 | sigmoid+xavier | on | 1.46 ± 0.0698 | 1.18 ± 0.15 | 1.15 ± 0.117 | 1.29 ± 0.269 |
| resnet | 32 | sigmoid+xavier | on | 1.48 ± 0.0332 | 1.13 ± 0.117 | 1.32 ± 0.0867 | 1.55 ± 0.257 |

## stem median r / eps32 (r = ||W_t+1 - W_t|| / ||W_t||, epoch median)

| arch | L | act/init | BN | epoch 0 | epoch 1 | epoch 5 | epoch 10 |
|---|---|---|---|---|---|---|---|
| plain | 8 | relu+kaiming | off | - | 8.07e+04 ± 6.53e+03 | 8.17e+04 ± 2.66e+03 | 9.51e+04 ± 2.76e+03 |
| resnet | 8 | relu+kaiming | off | - | n/a | n/a | n/a |
| plain | 16 | relu+kaiming | off | - | 5.97e+04 ± 1.04e+04 | 6.71e+04 ± 812 | 8.45e+04 ± 2.43e+03 |
| resnet | 16 | relu+kaiming | off | - | n/a | n/a | n/a |
| plain | 32 | relu+kaiming | off | - | 2.54e+04 ± 5.65e+03 (n=2) | 4.34e+04 ± 4.73e+03 (n=2) | 5.35e+04 ± 2.4e+03 (n=2) |
| resnet | 32 | relu+kaiming | off | - | n/a | n/a | n/a |
| plain | 8 | relu+kaiming | on | - | 1.16e+05 ± 2.11e+03 | 7.83e+04 ± 556 | 8.88e+04 ± 2.36e+03 |
| resnet | 8 | relu+kaiming | on | - | 1.3e+05 ± 4.33e+03 | 8.83e+04 ± 1.55e+03 | 9.97e+04 ± 3.91e+03 |
| plain | 16 | relu+kaiming | on | - | 5.3e+04 ± 1.01e+04 | 3.51e+04 ± 1.74e+03 | 5.01e+04 ± 2.73e+03 |
| resnet | 16 | relu+kaiming | on | - | 9.4e+04 ± 5.95e+03 | 6.48e+04 ± 1.28e+03 | 8.39e+04 ± 1.9e+03 |
| plain | 32 | relu+kaiming | on | - | 1.17e+03 ± 408 | 448 ± 88.4 | 423 ± 76.4 |
| resnet | 32 | relu+kaiming | on | - | 6.23e+04 ± 5.31e+03 | 3.81e+04 ± 2.87e+03 | 5.08e+04 ± 4.6e+03 |
| plain | 8 | sigmoid+xavier | off | - | 0.628 ± 0.0359 | 0.621 ± 0.0343 | 0.607 ± 0.0296 |
| resnet | 8 | sigmoid+xavier | off | - | 168 ± 4.97 | 175 ± 6.81 | 175 ± 4.59 |
| plain | 16 | sigmoid+xavier | off | - | 3.46e-08 ± 4.82e-09 | 3.37e-08 ± 2.97e-09 | 3.28e-08 ± 4.28e-09 |
| resnet | 16 | sigmoid+xavier | off | - | 0.461 ± 0.00583 | 0.485 ± 0.00784 | 0.491 ± 0.00938 |
| plain | 32 | sigmoid+xavier | off | - | 0 ± 0 | 0 ± 0 | 0 ± 0 |
| resnet | 32 | sigmoid+xavier | off | - | 1.39e-08 ± 7.64e-09 | 1.55e-08 ± 7.36e-09 | 1.42e-08 ± 8.22e-09 |
| plain | 8 | sigmoid+xavier | on | - | 4.77e+04 ± 795 | 6.12e+04 ± 794 | 7.3e+04 ± 996 |
| resnet | 8 | sigmoid+xavier | on | - | 4.64e+04 ± 826 | 6.27e+04 ± 1.02e+03 | 7.46e+04 ± 1.07e+03 |
| plain | 16 | sigmoid+xavier | on | - | 5.34e+04 ± 239 | 5.99e+04 ± 377 | 7.13e+04 ± 799 |
| resnet | 16 | sigmoid+xavier | on | - | 5.57e+04 ± 912 | 6.37e+04 ± 352 | 7.34e+04 ± 956 |
| plain | 32 | sigmoid+xavier | on | - | 5.34e+04 ± 391 | 5.6e+04 ± 862 | 6.38e+04 ± 2.8e+03 |
| resnet | 32 | sigmoid+xavier | on | - | 5.83e+04 ± 232 | 5.92e+04 ± 896 | 6.66e+04 ± 1.22e+03 |

## stem fraction of steps with W unchanged (torch.equal)

| arch | L | act/init | BN | epoch 0 | epoch 1 | epoch 5 | epoch 10 |
|---|---|---|---|---|---|---|---|
| plain | 8 | relu+kaiming | off | - | 0 ± 0 | 0 ± 0 | 0 ± 0 |
| resnet | 8 | relu+kaiming | off | - | n/a | n/a | n/a |
| plain | 16 | relu+kaiming | off | - | 0 ± 0 | 0 ± 0 | 0 ± 0 |
| resnet | 16 | relu+kaiming | off | - | n/a | n/a | n/a |
| plain | 32 | relu+kaiming | off | - | 0 ± 0 (n=2) | 0 ± 0 (n=2) | 0 ± 0 (n=2) |
| resnet | 32 | relu+kaiming | off | - | n/a | n/a | n/a |
| plain | 8 | relu+kaiming | on | - | 0 ± 0 | 0 ± 0 | 0 ± 0 |
| resnet | 8 | relu+kaiming | on | - | 0 ± 0 | 0 ± 0 | 0 ± 0 |
| plain | 16 | relu+kaiming | on | - | 0 ± 0 | 0 ± 0 | 0 ± 0 |
| resnet | 16 | relu+kaiming | on | - | 0 ± 0 | 0 ± 0 | 0 ± 0 |
| plain | 32 | relu+kaiming | on | - | 0 ± 0 | 0 ± 0 | 0 ± 0 |
| resnet | 32 | relu+kaiming | on | - | 0 ± 0 | 0 ± 0 | 0 ± 0 |
| plain | 8 | sigmoid+xavier | off | - | 0 ± 0 | 0 ± 0 | 0 ± 0 |
| resnet | 8 | sigmoid+xavier | off | - | 0 ± 0 | 0 ± 0 | 0 ± 0 |
| plain | 16 | sigmoid+xavier | off | - | 0 ± 0 | 0 ± 0 | 0 ± 0 |
| resnet | 16 | sigmoid+xavier | off | - | 0 ± 0 | 0 ± 0 | 0 ± 0 |
| plain | 32 | sigmoid+xavier | off | - | 1 ± 0 | 1 ± 0 | 1 ± 0 |
| resnet | 32 | sigmoid+xavier | off | - | 0.0127 ± 0.0219 | 0.00422 ± 0.00731 | 0.0127 ± 0.0127 |
| plain | 8 | sigmoid+xavier | on | - | 0 ± 0 | 0 ± 0 | 0 ± 0 |
| resnet | 8 | sigmoid+xavier | on | - | 0 ± 0 | 0 ± 0 | 0 ± 0 |
| plain | 16 | sigmoid+xavier | on | - | 0 ± 0 | 0 ± 0 | 0 ± 0 |
| resnet | 16 | sigmoid+xavier | on | - | 0 ± 0 | 0 ± 0 | 0 ± 0 |
| plain | 32 | sigmoid+xavier | on | - | 0 ± 0 | 0 ± 0 | 0 ± 0 |
| resnet | 32 | sigmoid+xavier | on | - | 0 ± 0 | 0 ± 0 | 0 ± 0 |

## test accuracy

| arch | L | act/init | BN | epoch 0 | epoch 1 | epoch 5 | epoch 10 |
|---|---|---|---|---|---|---|---|
| plain | 8 | relu+kaiming | off | 0.0982 ± 0.00245 | 0.236 ± 0.00734 | 0.329 ± 0.0331 | 0.345 ± 0.0151 |
| resnet | 8 | relu+kaiming | off | 0.104 ± 0.0077 | n/a | n/a | n/a |
| plain | 16 | relu+kaiming | off | 0.103 ± 0.00746 | 0.161 ± 0.0113 | 0.282 ± 0.054 | 0.321 ± 0.0184 |
| resnet | 16 | relu+kaiming | off | 0.103 ± 0.00509 | n/a | n/a | n/a |
| plain | 32 | relu+kaiming | off | 0.0994 ± 0.000954 | 0.115 ± 0.0257 (n=2) | 0.189 ± 0.000354 (n=2) | 0.161 ± 0.0682 (n=2) |
| resnet | 32 | relu+kaiming | off | 0.0958 ± 0.0037 | n/a | n/a | n/a |
| plain | 8 | relu+kaiming | on | 0.1 ± 0.00463 | 0.253 ± 0.00754 | 0.334 ± 0.0198 | 0.351 ± 0.041 |
| resnet | 8 | relu+kaiming | on | 0.0932 ± 0.00806 | 0.185 ± 0.016 | 0.251 ± 0.0468 | 0.316 ± 0.0572 |
| plain | 16 | relu+kaiming | on | 0.1 ± 0.000503 | 0.195 ± 0.0192 | 0.281 ± 0.0148 | 0.295 ± 0.0288 |
| resnet | 16 | relu+kaiming | on | 0.103 ± 0.00368 | 0.195 ± 0.0333 | 0.248 ± 0.0369 | 0.282 ± 0.0675 |
| plain | 32 | relu+kaiming | on | 0.106 ± 0.00962 | 0.131 ± 0.00649 | 0.177 ± 0.00396 | 0.229 ± 0.0109 |
| resnet | 32 | relu+kaiming | on | 0.1 ± 0.00052 | 0.183 ± 0.0403 | 0.258 ± 0.0431 | 0.32 ± 0.0577 |
| plain | 8 | sigmoid+xavier | off | 0.1 ± 1.7e-17 | 0.1 ± 1.7e-17 | 0.1 ± 1.7e-17 | 0.1 ± 1.7e-17 |
| resnet | 8 | sigmoid+xavier | off | 0.1 ± 1.7e-17 | 0.1 ± 1.7e-17 | 0.113 ± 0.0217 | 0.1 ± 1.7e-17 |
| plain | 16 | sigmoid+xavier | off | 0.1 ± 1.7e-17 | 0.1 ± 1.7e-17 | 0.1 ± 1.7e-17 | 0.1 ± 1.7e-17 |
| resnet | 16 | sigmoid+xavier | off | 0.1 ± 1.7e-17 | 0.1 ± 1.7e-17 | 0.1 ± 1.7e-17 | 0.1 ± 1.7e-17 |
| plain | 32 | sigmoid+xavier | off | 0.1 ± 1.7e-17 | 0.1 ± 1.7e-17 | 0.1 ± 1.7e-17 | 0.1 ± 1.7e-17 |
| resnet | 32 | sigmoid+xavier | off | 0.1 ± 1.7e-17 | 0.1 ± 1.7e-17 | 0.1 ± 1.7e-17 | 0.1 ± 1.7e-17 |
| plain | 8 | sigmoid+xavier | on | 0.1 ± 1.7e-17 | 0.229 ± 0.00698 | 0.258 ± 0.0295 | 0.251 ± 0.0375 |
| resnet | 8 | sigmoid+xavier | on | 0.1 ± 1.7e-17 | 0.215 ± 0.0336 | 0.223 ± 0.0332 | 0.265 ± 0.0536 |
| plain | 16 | sigmoid+xavier | on | 0.1 ± 1.7e-17 | 0.204 ± 0.02 | 0.211 ± 0.0433 | 0.245 ± 0.0527 |
| resnet | 16 | sigmoid+xavier | on | 0.1 ± 1.7e-17 | 0.193 ± 0.0114 | 0.173 ± 0.0168 | 0.262 ± 0.0677 |
| plain | 32 | sigmoid+xavier | on | 0.1 ± 1.7e-17 | 0.149 ± 0.0273 | 0.21 ± 0.0595 | 0.189 ± 0.0355 |
| resnet | 32 | sigmoid+xavier | on | 0.1 ± 1.7e-17 | 0.145 ± 0.0225 | 0.161 ± 0.0313 | 0.223 ± 0.0647 |

## diverged runs

| arch | L | act/init | BN | #diverged (of seeds) | where |
|---|---|---|---|---|---|
| plain | 8 | relu+kaiming | off | 0/3 | - |
| resnet | 8 | relu+kaiming | off | 3/3 | seed 42: epoch 1 step 3; seed 100: epoch 1 step 4; seed 2024: epoch 1 step 3 |
| plain | 16 | relu+kaiming | off | 0/3 | - |
| resnet | 16 | relu+kaiming | off | 3/3 | seed 42: epoch 1 step 2; seed 100: epoch 1 step 2; seed 2024: epoch 1 step 2 |
| plain | 32 | relu+kaiming | off | 1/3 | seed 42: epoch 1 step 3 |
| resnet | 32 | relu+kaiming | off | 3/3 | seed 42: epoch 1 step 1; seed 100: epoch 1 step 1; seed 2024: epoch 1 step 1 |
| plain | 8 | relu+kaiming | on | 0/3 | - |
| resnet | 8 | relu+kaiming | on | 0/3 | - |
| plain | 16 | relu+kaiming | on | 0/3 | - |
| resnet | 16 | relu+kaiming | on | 0/3 | - |
| plain | 32 | relu+kaiming | on | 0/3 | - |
| resnet | 32 | relu+kaiming | on | 0/3 | - |
| plain | 8 | sigmoid+xavier | off | 0/3 | - |
| resnet | 8 | sigmoid+xavier | off | 0/3 | - |
| plain | 16 | sigmoid+xavier | off | 0/3 | - |
| resnet | 16 | sigmoid+xavier | off | 0/3 | - |
| plain | 32 | sigmoid+xavier | off | 0/3 | - |
| resnet | 32 | sigmoid+xavier | off | 0/3 | - |
| plain | 8 | sigmoid+xavier | on | 0/3 | - |
| resnet | 8 | sigmoid+xavier | on | 0/3 | - |
| plain | 16 | sigmoid+xavier | on | 0/3 | - |
| resnet | 16 | sigmoid+xavier | on | 0/3 | - |
| plain | 32 | sigmoid+xavier | on | 0/3 | - |
| resnet | 32 | sigmoid+xavier | on | 0/3 | - |
