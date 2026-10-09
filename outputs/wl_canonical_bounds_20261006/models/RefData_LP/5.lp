\ Canonical explicit-bound formulation for WL comparison.
\ Finite bounds and singleton restrictions merged; original variable domains retained.
Maximize
  obj: + 80 x1 + 125 x2
Subject To
  Steel: + 8 x1 + 5 x2 <= 3500
  Iron: + 6 x1 + 4 x2 <= 1800
  MachineHours: + 4 x1 + 5 x2 <= 2800
  __wl_bound_00001_lower: + 1 x1 >= 0
  __wl_bound_00002_lower: + 1 x2 >= 0
Bounds
  x1 free
  x2 free
End
