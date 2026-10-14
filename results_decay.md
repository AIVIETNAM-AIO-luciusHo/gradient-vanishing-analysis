# Total Degradation (TD) and decay_slope from results_training.json (mean +/- std, ddof=1, over seeds)

`source: results_training.json (no new training) | seeds=[42, 100, 2024] | epochs [0, 1, 5, 10] | numpy 2.5.3`

- TD = log10(||g_first|| / ||g_last||) = log10(rho) of results_training.md (g_first = stem, g_last = head); TD < 0: attenuation toward the first layer, TD > 0: amplification.
- decay_slope = OLS slope of log10(||g^[l]||_F) vs layer index l = 0..L+1 (stem, blocks, head; L+2 points), in orders of magnitude per layer; slope > 0: gradient grows toward the head (vanishing toward the stem). R^2 is the R^2 of the same regression.
- gnorm = 0 or non-finite entries are dropped from the regression (count in the last column); fewer than 3 points -> n/a. Diverged runs are skipped; (n=k) = only k seeds not diverged. TD column equals the log10(rho) table of results_training.md cell by cell (asserted).

| arch | L | act/init | BN | epoch | n runs | TD = log10(rho) | decay_slope | R^2 | dropped gnorm entries |
|---|---|---|---|---|---|---|---|---|---|
| plain | 8 | relu+kaiming | off | 0 | 3 | -0.0372 ± 0.0977 | 0.0282 ± 0.0146 | 0.354 ± 0.233 | 0 |
| plain | 8 | relu+kaiming | off | 1 | 3 | 0.0948 ± 0.172 | 0.00131 ± 0.0176 | 0.186 ± 0.186 | 0 |
| plain | 8 | relu+kaiming | off | 5 | 3 | 0.222 ± 0.121 | -0.0202 ± 0.0113 | 0.295 ± 0.191 | 0 |
| plain | 8 | relu+kaiming | off | 10 | 3 | 0.328 ± 0.0488 | -0.0367 ± 0.00694 | 0.691 ± 0.118 | 0 |
| plain | 8 | relu+kaiming | on | 0 | 3 | 0.862 ± 0.0482 | -0.0841 ± 0.00268 | 0.604 ± 0.035 | 0 |
| plain | 8 | relu+kaiming | on | 1 | 3 | 0.0232 ± 0.00475 | -0.0116 ± 0.000307 | 0.0788 ± 0.0231 | 0 |
| plain | 8 | relu+kaiming | on | 5 | 3 | 0.187 ± 0.129 | -0.0294 ± 0.00814 | 0.399 ± 0.258 | 0 |
| plain | 8 | relu+kaiming | on | 10 | 3 | 0.332 ± 0.0716 | -0.0407 ± 0.0076 | 0.722 ± 0.0816 | 0 |
| plain | 8 | sigmoid+xavier | off | 0 | 3 | -5.05 ± 0.0904 | 0.596 ± 0.00406 | 0.995 ± 0.0025 | 0 |
| plain | 8 | sigmoid+xavier | off | 1 | 3 | -5.16 ± 0.114 | 0.604 ± 0.00736 | 0.997 ± 0.00158 | 0 |
| plain | 8 | sigmoid+xavier | off | 5 | 3 | -5.02 ± 0.0995 | 0.593 ± 0.00587 | 0.995 ± 0.00223 | 0 |
| plain | 8 | sigmoid+xavier | off | 10 | 3 | -4.82 ± 0.105 | 0.58 ± 0.00519 | 0.992 ± 0.00324 | 0 |
| plain | 8 | sigmoid+xavier | on | 0 | 3 | -0.16 ± 0.0756 | 0.00182 ± 0.00506 | 0.00323 ± 0.00289 | 0 |
| plain | 8 | sigmoid+xavier | on | 1 | 3 | -0.196 ± 0.173 | 0.00691 ± 0.011 | 0.0159 ± 0.0265 | 0 |
| plain | 8 | sigmoid+xavier | on | 5 | 3 | -0.00282 ± 0.127 | -0.0102 ± 0.0108 | 0.0663 ± 0.105 | 0 |
| plain | 8 | sigmoid+xavier | on | 10 | 3 | 0.0941 ± 0.122 | -0.0192 ± 0.0114 | 0.182 ± 0.161 | 0 |
| plain | 16 | relu+kaiming | off | 0 | 3 | 0.0626 ± 0.139 | 0.0154 ± 0.00474 | 0.424 ± 0.223 | 0 |
| plain | 16 | relu+kaiming | off | 1 | 3 | 0.121 ± 0.0846 | -0.000382 ± 0.00682 | 0.102 ± 0.0714 | 0 |
| plain | 16 | relu+kaiming | off | 5 | 3 | 0.342 ± 0.141 | -0.0155 ± 0.00484 | 0.571 ± 0.127 | 0 |
| plain | 16 | relu+kaiming | off | 10 | 3 | 0.305 ± 0.0909 | -0.0201 ± 0.00472 | 0.556 ± 0.165 | 0 |
| plain | 16 | relu+kaiming | on | 0 | 3 | 1.56 ± 0.0611 | -0.0826 ± 0.00112 | 0.891 ± 0.0123 | 0 |
| plain | 16 | relu+kaiming | on | 1 | 3 | -0.297 ± 0.0768 | 0.00536 ± 0.00369 | 0.176 ± 0.171 | 0 |
| plain | 16 | relu+kaiming | on | 5 | 3 | -0.247 ± 0.0737 | -0.0019 ± 0.00413 | 0.0374 ± 0.0598 | 0 |
| plain | 16 | relu+kaiming | on | 10 | 3 | -0.0106 ± 0.144 | -0.0117 ± 0.00607 | 0.259 ± 0.251 | 0 |
| plain | 16 | sigmoid+xavier | off | 0 | 3 | -10.1 ± 0.126 | 0.62 ± 0.0033 | 0.999 ± 0.000351 | 0 |
| plain | 16 | sigmoid+xavier | off | 1 | 3 | -10.2 ± 0.181 | 0.619 ± 0.00176 | 0.999 ± 0.000657 | 0 |
| plain | 16 | sigmoid+xavier | off | 5 | 3 | -10 ± 0.0824 | 0.619 ± 0.00551 | 0.999 ± 0.000576 | 0 |
| plain | 16 | sigmoid+xavier | off | 10 | 3 | -9.84 ± 0.146 | 0.613 ± 0.00665 | 0.998 ± 0.000307 | 0 |
| plain | 16 | sigmoid+xavier | on | 0 | 3 | -0.0732 ± 0.119 | -0.00524 ± 0.00203 | 0.0314 ± 0.031 | 0 |
| plain | 16 | sigmoid+xavier | on | 1 | 3 | -0.165 ± 0.184 | 0.000832 ± 0.00464 | 0.0142 ± 0.0138 | 0 |
| plain | 16 | sigmoid+xavier | on | 5 | 3 | -0.00444 ± 0.133 | -0.00572 ± 0.00481 | 0.0954 ± 0.134 | 0 |
| plain | 16 | sigmoid+xavier | on | 10 | 3 | 0.103 ± 0.143 | -0.00925 ± 0.00447 | 0.224 ± 0.18 | 0 |
| plain | 32 | relu+kaiming | off | 0 | 3 | -0.133 ± 0.191 | 0.0111 ± 0.00588 | 0.576 ± 0.203 | 0 |
| plain | 32 | relu+kaiming | off | 1 | 2 | 0.221 ± 0.317 (n=2) | 0.00372 ± 0.00125 (n=2) | 0.149 ± 0.00321 (n=2) | 0 |
| plain | 32 | relu+kaiming | off | 5 | 2 | 0.384 ± 0.0145 (n=2) | -0.00321 ± 0.00199 (n=2) | 0.0779 ± 0.0646 (n=2) | 0 |
| plain | 32 | relu+kaiming | off | 10 | 2 | 0.0497 ± 0.545 (n=2) | 0.000233 ± 0.00769 (n=2) | 0.38 ± 0.199 (n=2) | 0 |
| plain | 32 | relu+kaiming | on | 0 | 3 | 2.81 ± 0.0832 | -0.0792 ± 0.00161 | 0.978 ± 0.00236 | 0 |
| plain | 32 | relu+kaiming | on | 1 | 3 | -1.3 ± 0.0419 | 0.0235 ± 0.00159 | 0.779 ± 0.0543 | 0 |
| plain | 32 | relu+kaiming | on | 5 | 3 | -1.36 ± 0.154 | 0.0138 ± 0.00261 | 0.445 ± 0.0349 | 0 |
| plain | 32 | relu+kaiming | on | 10 | 3 | -1.26 ± 0.145 | 0.00797 ± 0.00121 | 0.196 ± 0.0597 | 0 |
| plain | 32 | sigmoid+xavier | off | 0 | 3 | -20.5 ± 0.101 | 0.627 ± 0.00313 | 1 ± 3.31e-05 | 0 |
| plain | 32 | sigmoid+xavier | off | 1 | 3 | -20.4 ± 0.131 | 0.627 ± 0.000472 | 1 ± 2.88e-05 | 0 |
| plain | 32 | sigmoid+xavier | off | 5 | 3 | -20.3 ± 0.161 | 0.623 ± 0.00223 | 1 ± 4.36e-05 | 0 |
| plain | 32 | sigmoid+xavier | off | 10 | 3 | -20.1 ± 0.132 | 0.624 ± 0.00149 | 1 ± 8.52e-05 | 0 |
| plain | 32 | sigmoid+xavier | on | 0 | 3 | -0.0524 ± 0.0885 | -0.00678 ± 0.000826 | 0.177 ± 0.0408 | 0 |
| plain | 32 | sigmoid+xavier | on | 1 | 3 | -0.142 ± 0.147 | -0.0014 ± 0.000646 | 0.0165 ± 0.0118 | 0 |
| plain | 32 | sigmoid+xavier | on | 5 | 3 | -0.0711 ± 0.13 | -0.00217 ± 0.000754 | 0.0533 ± 0.0425 | 0 |
| plain | 32 | sigmoid+xavier | on | 10 | 3 | 0.0398 ± 0.157 | -0.00385 ± 0.00171 | 0.186 ± 0.163 | 0 |
| resnet | 8 | relu+kaiming | off | 0 | 3 | -0.279 ± 0.174 | 0.056 ± 0.0139 | 0.605 ± 0.217 | 0 |
| resnet | 8 | relu+kaiming | off | 1 | 0 | n/a | n/a | n/a | 0 |
| resnet | 8 | relu+kaiming | off | 5 | 0 | n/a | n/a | n/a | 0 |
| resnet | 8 | relu+kaiming | off | 10 | 0 | n/a | n/a | n/a | 0 |
| resnet | 8 | relu+kaiming | on | 0 | 3 | 0.2 ± 0.0665 | -0.0331 ± 0.00459 | 0.0875 ± 0.0279 | 0 |
| resnet | 8 | relu+kaiming | on | 1 | 3 | -0.674 ± 0.0702 | 0.0312 ± 0.00435 | 0.106 ± 0.0182 | 0 |
| resnet | 8 | relu+kaiming | on | 5 | 3 | -0.415 ± 0.0941 | 0.0121 ± 0.00526 | 0.0202 ± 0.0138 | 0 |
| resnet | 8 | relu+kaiming | on | 10 | 3 | -0.122 ± 0.0954 | -0.00937 ± 0.0109 | 0.0375 ± 0.0359 | 0 |
| resnet | 8 | sigmoid+xavier | off | 0 | 3 | -2.83 ± 0.106 | 0.327 ± 0.00717 | 0.946 ± 0.014 | 0 |
| resnet | 8 | sigmoid+xavier | off | 1 | 3 | -2.99 ± 0.151 | 0.338 ± 0.00796 | 0.966 ± 0.0128 | 0 |
| resnet | 8 | sigmoid+xavier | off | 5 | 3 | -2.8 ± 0.0838 | 0.325 ± 0.00463 | 0.957 ± 0.00807 | 0 |
| resnet | 8 | sigmoid+xavier | off | 10 | 3 | -2.57 ± 0.126 | 0.312 ± 0.00695 | 0.942 ± 0.0188 | 0 |
| resnet | 8 | sigmoid+xavier | on | 0 | 3 | -0.347 ± 0.0787 | 0.0115 ± 0.00586 | 0.0158 ± 0.0138 | 0 |
| resnet | 8 | sigmoid+xavier | on | 1 | 3 | -0.443 ± 0.2 | 0.0228 ± 0.014 | 0.064 ± 0.0527 | 0 |
| resnet | 8 | sigmoid+xavier | on | 5 | 3 | -0.215 ± 0.137 | 0.00362 ± 0.00762 | 0.0118 ± 0.00973 | 0 |
| resnet | 8 | sigmoid+xavier | on | 10 | 3 | -0.0564 ± 0.149 | -0.00987 ± 0.0126 | 0.0793 ± 0.0706 | 0 |
| resnet | 16 | relu+kaiming | off | 0 | 3 | -0.491 ± 0.157 | 0.0417 ± 0.00558 | 0.828 ± 0.0725 | 0 |
| resnet | 16 | relu+kaiming | off | 1 | 0 | n/a | n/a | n/a | 0 |
| resnet | 16 | relu+kaiming | off | 5 | 0 | n/a | n/a | n/a | 0 |
| resnet | 16 | relu+kaiming | off | 10 | 0 | n/a | n/a | n/a | 0 |
| resnet | 16 | relu+kaiming | on | 0 | 3 | 0.3 ± 0.0452 | -0.0302 ± 0.00143 | 0.222 ± 0.00444 | 0 |
| resnet | 16 | relu+kaiming | on | 1 | 3 | -0.731 ± 0.184 | 0.00698 ± 0.00399 | 0.0263 ± 0.0205 | 0 |
| resnet | 16 | relu+kaiming | on | 5 | 3 | -0.488 ± 0.0698 | -0.00507 ± 0.0031 | 0.0153 ± 0.0126 | 0 |
| resnet | 16 | relu+kaiming | on | 10 | 3 | -0.25 ± 0.0925 | -0.0107 ± 0.00189 | 0.0747 ± 0.0327 | 0 |
| resnet | 16 | sigmoid+xavier | off | 0 | 3 | -5.4 ± 0.11 | 0.324 ± 0.00124 | 0.988 ± 0.0017 | 0 |
| resnet | 16 | sigmoid+xavier | off | 1 | 3 | -5.55 ± 0.161 | 0.327 ± 0.00347 | 0.991 ± 0.00245 | 0 |
| resnet | 16 | sigmoid+xavier | off | 5 | 3 | -5.37 ± 0.0814 | 0.324 ± 0.000676 | 0.989 ± 0.00242 | 0 |
| resnet | 16 | sigmoid+xavier | off | 10 | 3 | -5.13 ± 0.147 | 0.319 ± 0.00255 | 0.987 ± 0.0031 | 0 |
| resnet | 16 | sigmoid+xavier | on | 0 | 3 | -0.265 ± 0.109 | -0.003 ± 0.00298 | 0.0097 ± 0.0143 | 0 |
| resnet | 16 | sigmoid+xavier | on | 1 | 3 | -0.425 ± 0.213 | 0.00586 ± 0.00541 | 0.0316 ± 0.0314 | 0 |
| resnet | 16 | sigmoid+xavier | on | 5 | 3 | -0.203 ± 0.148 | -0.00399 ± 0.00521 | 0.0506 ± 0.0805 | 0 |
| resnet | 16 | sigmoid+xavier | on | 10 | 3 | -0.0271 ± 0.104 | -0.0104 ± 0.0041 | 0.165 ± 0.119 | 0 |
| resnet | 32 | relu+kaiming | off | 0 | 3 | -0.737 ± 0.0462 | 0.0293 ± 0.00214 | 0.934 ± 0.015 | 0 |
| resnet | 32 | relu+kaiming | off | 1 | 0 | n/a | n/a | n/a | 0 |
| resnet | 32 | relu+kaiming | off | 5 | 0 | n/a | n/a | n/a | 0 |
| resnet | 32 | relu+kaiming | off | 10 | 0 | n/a | n/a | n/a | 0 |
| resnet | 32 | relu+kaiming | on | 0 | 3 | 0.348 ± 0.101 | -0.0239 ± 0.000603 | 0.402 ± 0.0194 | 0 |
| resnet | 32 | relu+kaiming | on | 1 | 3 | -0.914 ± 0.0523 | 0.0023 ± 0.00109 | 0.0165 ± 0.0142 | 0 |
| resnet | 32 | relu+kaiming | on | 5 | 3 | -0.569 ± 0.145 | -0.00533 ± 0.00352 | 0.075 ± 0.0739 | 0 |
| resnet | 32 | relu+kaiming | on | 10 | 3 | -0.342 ± 0.156 | -0.00728 ± 0.00231 | 0.127 ± 0.0757 | 0 |
| resnet | 32 | sigmoid+xavier | off | 0 | 3 | -10.8 ± 0.103 | 0.325 ± 0.000483 | 0.998 ± 0.000174 | 0 |
| resnet | 32 | sigmoid+xavier | off | 1 | 3 | -10.8 ± 0.128 | 0.326 ± 0.00108 | 0.998 ± 0.000303 | 0 |
| resnet | 32 | sigmoid+xavier | off | 5 | 3 | -10.6 ± 0.101 | 0.325 ± 0.000463 | 0.998 ± 0.000426 | 0 |
| resnet | 32 | sigmoid+xavier | off | 10 | 3 | -10.3 ± 0.0984 | 0.325 ± 0.00241 | 0.997 ± 0.000608 | 0 |
| resnet | 32 | sigmoid+xavier | on | 0 | 3 | -0.242 ± 0.078 | -0.00732 ± 0.00112 | 0.131 ± 0.0178 | 0 |
| resnet | 32 | sigmoid+xavier | on | 1 | 3 | -0.409 ± 0.196 | -0.000647 ± 0.00157 | 0.0148 ± 0.0252 | 0 |
| resnet | 32 | sigmoid+xavier | on | 5 | 3 | -0.227 ± 0.142 | -0.00414 ± 0.00177 | 0.113 ± 0.111 | 0 |
| resnet | 32 | sigmoid+xavier | on | 10 | 3 | -0.0447 ± 0.131 | -0.0073 ± 0.00171 | 0.324 ± 0.173 | 0 |
