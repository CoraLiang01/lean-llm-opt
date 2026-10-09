\ Canonical explicit-bound formulation for WL comparison.
\ Finite bounds and singleton restrictions merged; original variable domains retained.
Minimize
  obj: + 2358.3899999999999 x_0_0 + 1492.0799999999999 x_0_1 + 0.070000000000000007 x_1_0 + 52.32 x_1_1 + 105.97 y_0 + 85.310000000000002 y_1
Subject To
  R0: + 1 x_0_0 + 1 x_1_0 = 144
  R1: + 1 x_0_1 + 1 x_1_1 = 216
  R2: + 1 x_0_0 + 1 x_0_1 - 360 y_0 <= 0
  R3: + 1 x_1_0 + 1 x_1_1 - 360 y_1 <= 0
  __wl_bound_00001_lower: + 1 x_0_0 >= 0
  __wl_bound_00002_lower: + 1 x_0_1 >= 0
  __wl_bound_00003_lower: + 1 x_1_0 >= 0
  __wl_bound_00004_lower: + 1 x_1_1 >= 0
  __wl_bound_00005_lower: + 1 y_0 >= 0
  __wl_bound_00005_upper: + 1 y_0 <= 1
  __wl_bound_00006_lower: + 1 y_1 >= 0
  __wl_bound_00006_upper: + 1 y_1 <= 1
Bounds
  x_0_0 free
  x_0_1 free
  x_1_0 free
  x_1_1 free
Binaries
  y_0 y_1
End
