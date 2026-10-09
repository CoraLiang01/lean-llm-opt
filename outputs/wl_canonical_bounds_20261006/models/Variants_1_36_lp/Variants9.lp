\ Canonical explicit-bound formulation for WL comparison.
\ Finite bounds and singleton restrictions merged; original variable domains retained.
Minimize
  obj: + 1 y_P1 + 1 y_P2 + 1 y_P3 + 1 y_P4 + 1 y_P5 + 1 y_P6 + 1 y_P7 + 1 y_P8 + 1 y_P9
Subject To
  demand_A: + 4 y_P1 + 2 y_P5 + 1 y_P6 + 1 y_P8 + 2 y_P9 >= 24
  demand_B: + 3 y_P2 + 1 y_P5 + 1 y_P7 + 1 y_P8 >= 18
  demand_C: + 2 y_P3 + 1 y_P6 + 1 y_P8 >= 12
  demand_D: + 2 y_P4 + 1 y_P7 + 1 y_P9 >= 10
  __wl_bound_00001_lower: + 1 y_P1 >= 0
  __wl_bound_00002_lower: + 1 y_P2 >= 0
  __wl_bound_00003_lower: + 1 y_P3 >= 0
  __wl_bound_00004_lower: + 1 y_P4 >= 0
  __wl_bound_00005_lower: + 1 y_P5 >= 0
  __wl_bound_00006_lower: + 1 y_P6 >= 0
  __wl_bound_00007_lower: + 1 y_P7 >= 0
  __wl_bound_00008_lower: + 1 y_P8 >= 0
  __wl_bound_00009_lower: + 1 y_P9 >= 0
Bounds
  y_P1 free
  y_P2 free
  y_P3 free
  y_P4 free
  y_P5 free
  y_P6 free
  y_P7 free
  y_P8 free
  y_P9 free
Generals
  y_P1 y_P2 y_P3 y_P4 y_P5 y_P6 y_P7 y_P8 y_P9
End
