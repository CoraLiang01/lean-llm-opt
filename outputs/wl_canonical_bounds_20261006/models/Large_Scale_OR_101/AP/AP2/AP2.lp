\ Canonical explicit-bound formulation for WL comparison.
\ Finite bounds and singleton restrictions merged; original variable domains retained.
Minimize
  obj: + 3000 x_0_0 + 3200 x_0_1 + 3100 x_0_2 + 2800 x_1_0 + 3300 x_1_1 + 2900 x_1_2 + 2900 x_2_0 + 3100 x_2_1 + 3000 x_2_2
Subject To
  R0: + 1 x_0_0 + 1 x_0_1 + 1 x_0_2 = 1
  R1: + 1 x_1_0 + 1 x_1_1 + 1 x_1_2 = 1
  R2: + 1 x_2_0 + 1 x_2_1 + 1 x_2_2 = 1
  R3: + 1 x_0_0 + 1 x_1_0 + 1 x_2_0 = 1
  R4: + 1 x_0_1 + 1 x_1_1 + 1 x_2_1 = 1
  R5: + 1 x_0_2 + 1 x_1_2 + 1 x_2_2 = 1
  __wl_bound_00001_lower: + 1 x_0_0 >= 0
  __wl_bound_00002_lower: + 1 x_0_1 >= 0
  __wl_bound_00003_lower: + 1 x_0_2 >= 0
  __wl_bound_00004_lower: + 1 x_1_0 >= 0
  __wl_bound_00005_lower: + 1 x_1_1 >= 0
  __wl_bound_00006_lower: + 1 x_1_2 >= 0
  __wl_bound_00007_lower: + 1 x_2_0 >= 0
  __wl_bound_00008_lower: + 1 x_2_1 >= 0
  __wl_bound_00009_lower: + 1 x_2_2 >= 0
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
End
