\ Canonical explicit-bound formulation for WL comparison.
\ Finite bounds and singleton restrictions merged; original variable domains retained.
Maximize
  obj: + 95 x_1 + 92 x_2 + 86 x_3 + 82 x_4 + 85 x_5 + 80 x_6 + 88 x_7
Subject To
  credit_limit: + 5 x_1 + 5 x_2 + 4 x_3 + 4 x_4 + 4 x_5 + 4 x_6 + 4 x_7 <= 20
  __wl_bound_00001_lower: + 1 x_1 >= 0
  __wl_bound_00001_upper: + 1 x_1 <= 1
  __wl_bound_00002_lower: + 1 x_2 >= 0
  __wl_bound_00002_upper: + 1 x_2 <= 1
  __wl_bound_00003_lower: + 1 x_3 >= 0
  __wl_bound_00003_upper: + 1 x_3 <= 1
  __wl_bound_00004_lower: + 1 x_4 >= 0
  __wl_bound_00004_upper: + 1 x_4 <= 1
  __wl_bound_00005_lower: + 1 x_5 >= 0
  __wl_bound_00005_upper: + 1 x_5 <= 1
  __wl_bound_00006_lower: + 1 x_6 >= 0
  __wl_bound_00006_upper: + 1 x_6 <= 1
  __wl_bound_00007_lower: + 1 x_7 >= 0
  __wl_bound_00007_upper: + 1 x_7 <= 1
Bounds
Binaries
  x_1 x_2 x_3 x_4 x_5 x_6 x_7
End
