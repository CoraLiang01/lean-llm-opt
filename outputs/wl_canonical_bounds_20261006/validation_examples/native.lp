\ Canonical explicit-bound formulation for WL comparison.
\ Finite bounds and singleton restrictions merged; original variable domains retained.
Minimize
  obj: 0 x
Subject To
  coupling: + 1 x + 1 y <= 5
  __wl_bound_00001_lower: + 1 x >= 0
  __wl_bound_00001_upper: + 1 x <= 3
  __wl_bound_00002_lower: + 1 y >= 0
Bounds
  x free
  y free
End
