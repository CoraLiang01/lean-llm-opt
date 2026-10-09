\ Canonical explicit-bound formulation for WL comparison.
\ Finite bounds and singleton restrictions merged; original variable domains retained.
Minimize
  obj: + 1506.22 x_0_0 + 70.900000000000006 x_0_1 + 8.4399999999999995 x_0_2 + 1732.6500000000001 x_1_0 + 1780.72 x_1_1 + 567.44000000000005 x_1_2 + 115.66 x_2_0 + 100.76000000000001 x_2_1 + 64.680000000000007 x_2_2 + 102.33 y_0 + 94.920000000000002 y_1 + 91.829999999999998 y_2
Subject To
  R0: + 1 x_0_0 + 1 x_1_0 + 1 x_2_0 = 1083
  R1: + 1 x_0_1 + 1 x_1_1 + 1 x_2_1 = 776
  R2: + 1 x_0_2 + 1 x_1_2 + 1 x_2_2 = 16214
  R3: + 1 x_0_0 + 1 x_0_1 + 1 x_0_2 - 18073 y_0 <= 0
  R4: + 1 x_1_0 + 1 x_1_1 + 1 x_1_2 - 18073 y_1 <= 0
  R5: + 1 x_2_0 + 1 x_2_1 + 1 x_2_2 - 18073 y_2 <= 0
  __wl_bound_00001_lower: + 1 x_0_0 >= 0
  __wl_bound_00002_lower: + 1 x_0_1 >= 0
  __wl_bound_00003_lower: + 1 x_0_2 >= 0
  __wl_bound_00004_lower: + 1 x_1_0 >= 0
  __wl_bound_00005_lower: + 1 x_1_1 >= 0
  __wl_bound_00006_lower: + 1 x_1_2 >= 0
  __wl_bound_00007_lower: + 1 x_2_0 >= 0
  __wl_bound_00008_lower: + 1 x_2_1 >= 0
  __wl_bound_00009_lower: + 1 x_2_2 >= 0
  __wl_bound_00010_lower: + 1 y_0 >= 0
  __wl_bound_00010_upper: + 1 y_0 <= 1
  __wl_bound_00011_lower: + 1 y_1 >= 0
  __wl_bound_00011_upper: + 1 y_1 <= 1
  __wl_bound_00012_lower: + 1 y_2 >= 0
  __wl_bound_00012_upper: + 1 y_2 <= 1
Bounds
  x_0_0 free
  x_0_1 free
  x_0_2 free
  x_1_0 free
  x_1_1 free
  x_1_2 free
  x_2_0 free
  x_2_1 free
  x_2_2 free
Binaries
  y_0 y_1 y_2
End
