\ Canonical native-bound formulation; bounds excluded from WL graph.
Minimize
  obj: + 3000 x_0_0 + 3200 x_0_1 + 3100 x_0_2 + 2800 x_1_0 + 3300 x_1_1 + 2900 x_1_2 + 2900 x_2_0 + 3100 x_2_1 + 3000 x_2_2
Subject To
  R0: + 1 x_0_0 + 1 x_0_1 + 1 x_0_2 = 1
  R1: + 1 x_1_0 + 1 x_1_1 + 1 x_1_2 = 1
  R2: + 1 x_2_0 + 1 x_2_1 + 1 x_2_2 = 1
  R3: + 1 x_0_0 + 1 x_1_0 + 1 x_2_0 = 1
  R4: + 1 x_0_1 + 1 x_1_1 + 1 x_2_1 = 1
  R5: + 1 x_0_2 + 1 x_1_2 + 1 x_2_2 = 1
Bounds
  0 <= x_0_0 <= +inf
  0 <= x_0_1 <= +inf
  0 <= x_0_2 <= +inf
  0 <= x_1_0 <= +inf
  0 <= x_1_1 <= +inf
  0 <= x_1_2 <= +inf
  0 <= x_2_0 <= +inf
  0 <= x_2_1 <= +inf
  0 <= x_2_2 <= +inf
End
