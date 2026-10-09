\ Canonical explicit-bound formulation for WL comparison.
\ Finite bounds and singleton restrictions merged; original variable domains retained.
Maximize
  obj: + 56 x_1 + 21.600000000000001 x_2 + 62.5 x_3
Subject To
  __wl_bound_00001_lower: + 1 x_1 >= 0
  __wl_bound_00001_upper: + 1 x_1 <= 5
  __wl_bound_00002_lower: + 1 x_2 >= 0
  __wl_bound_00002_upper: + 1 x_2 <= 3
  __wl_bound_00003_lower: + 1 x_3 >= 0
  __wl_bound_00003_upper: + 1 x_3 <= 3
Bounds
  x_1 free
  x_2 free
  x_3 free
Generals
  x_1 x_2 x_3
End
