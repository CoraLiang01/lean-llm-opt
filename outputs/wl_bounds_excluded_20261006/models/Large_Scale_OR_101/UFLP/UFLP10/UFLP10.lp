\ Canonical native-bound formulation; bounds excluded from WL graph.
Minimize
  obj: + 2358.3899999999999 x_0_0 + 1492.0799999999999 x_0_1 + 0.070000000000000007 x_1_0 + 52.32 x_1_1 + 105.97 y_0 + 85.310000000000002 y_1
Subject To
  R0: + 1 x_0_0 + 1 x_1_0 = 144
  R1: + 1 x_0_1 + 1 x_1_1 = 216
  R2: + 1 x_0_0 + 1 x_0_1 - 360 y_0 <= 0
  R3: + 1 x_1_0 + 1 x_1_1 - 360 y_1 <= 0
Bounds
  0 <= x_0_0 <= +inf
  0 <= x_0_1 <= +inf
  0 <= x_1_0 <= +inf
  0 <= x_1_1 <= +inf
  0 <= y_0 <= 1
  0 <= y_1 <= 1
Binaries
  y_0 y_1
End
