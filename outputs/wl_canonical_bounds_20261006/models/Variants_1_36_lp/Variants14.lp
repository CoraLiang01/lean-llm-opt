\ Canonical explicit-bound formulation for WL comparison.
\ Finite bounds and singleton restrictions merged; original variable domains retained.
Minimize
  obj: + 11 y_B1 + 14 y_B2 + 10 y_B3 + 13 y_B4 + 16 y_B5 + 9 y_B6 + 12 y_B7 + 15 y_B8
Subject To
  cover_Z1: + 1 y_B1 + 1 y_B4 >= 1
  cover_Z2: + 1 y_B1 + 1 y_B2 >= 1
  cover_Z3: + 1 y_B2 + 1 y_B5 >= 1
  cover_Z4: + 1 y_B3 + 1 y_B7 >= 1
  cover_Z5: + 1 y_B1 + 1 y_B3 + 1 y_B8 >= 1
  cover_Z6: + 1 y_B2 + 1 y_B4 + 1 y_B8 >= 1
  cover_Z7: + 1 y_B4 + 1 y_B5 >= 1
  cover_Z8: + 1 y_B3 + 1 y_B6 >= 1
  cover_Z9: + 1 y_B5 + 1 y_B6 + 1 y_B8 >= 1
  cover_Z10: + 1 y_B6 + 1 y_B7 >= 1
  __wl_bound_00001_lower: + 1 y_B1 >= 0
  __wl_bound_00001_upper: + 1 y_B1 <= 1
  __wl_bound_00002_lower: + 1 y_B2 >= 0
  __wl_bound_00002_upper: + 1 y_B2 <= 1
  __wl_bound_00003_lower: + 1 y_B3 >= 0
  __wl_bound_00003_upper: + 1 y_B3 <= 1
  __wl_bound_00004_lower: + 1 y_B4 >= 0
  __wl_bound_00004_upper: + 1 y_B4 <= 1
  __wl_bound_00005_lower: + 1 y_B5 >= 0
  __wl_bound_00005_upper: + 1 y_B5 <= 1
  __wl_bound_00006_lower: + 1 y_B6 >= 0
  __wl_bound_00006_upper: + 1 y_B6 <= 1
  __wl_bound_00007_lower: + 1 y_B7 >= 0
  __wl_bound_00007_upper: + 1 y_B7 <= 1
  __wl_bound_00008_lower: + 1 y_B8 >= 0
  __wl_bound_00008_upper: + 1 y_B8 <= 1
Bounds
Binaries
  y_B1 y_B2 y_B3 y_B4 y_B5 y_B6 y_B7 y_B8
End
