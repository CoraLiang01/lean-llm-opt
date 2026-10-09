\ Canonical explicit-bound formulation for WL comparison.
\ Finite bounds and singleton restrictions merged; original variable domains retained.
Minimize
  obj: + 12 y_SC1 + 15 y_SC2 + 18 y_SC3 + 10 y_SC4 + 14 y_SC5 + 13 y_SC6 + 16 y_SC7 + 11 y_SC8
Subject To
  cover_D1: + 1 y_SC1 + 1 y_SC7 >= 1
  cover_D2: + 1 y_SC1 + 1 y_SC2 >= 1
  cover_D3: + 1 y_SC2 + 1 y_SC8 >= 1
  cover_D4: + 1 y_SC1 + 1 y_SC3 + 1 y_SC8 >= 1
  cover_D5: + 1 y_SC2 + 1 y_SC3 >= 1
  cover_D6: + 1 y_SC3 + 1 y_SC4 >= 1
  cover_D7: + 1 y_SC4 + 1 y_SC5 >= 1
  cover_D8: + 1 y_SC5 + 1 y_SC6 + 1 y_SC8 >= 1
  cover_D9: + 1 y_SC6 + 1 y_SC7 >= 1
  cover_D10: + 1 y_SC5 + 1 y_SC7 >= 1
  __wl_bound_00001_lower: + 1 y_SC1 >= 0
  __wl_bound_00001_upper: + 1 y_SC1 <= 1
  __wl_bound_00002_lower: + 1 y_SC2 >= 0
  __wl_bound_00002_upper: + 1 y_SC2 <= 1
  __wl_bound_00003_lower: + 1 y_SC3 >= 0
  __wl_bound_00003_upper: + 1 y_SC3 <= 1
  __wl_bound_00004_lower: + 1 y_SC4 >= 0
  __wl_bound_00004_upper: + 1 y_SC4 <= 1
  __wl_bound_00005_lower: + 1 y_SC5 >= 0
  __wl_bound_00005_upper: + 1 y_SC5 <= 1
  __wl_bound_00006_lower: + 1 y_SC6 >= 0
  __wl_bound_00006_upper: + 1 y_SC6 <= 1
  __wl_bound_00007_lower: + 1 y_SC7 >= 0
  __wl_bound_00007_upper: + 1 y_SC7 <= 1
  __wl_bound_00008_lower: + 1 y_SC8 >= 0
  __wl_bound_00008_upper: + 1 y_SC8 <= 1
Bounds
Binaries
  y_SC1 y_SC2 y_SC3 y_SC4 y_SC5 y_SC6 y_SC7 y_SC8
End
