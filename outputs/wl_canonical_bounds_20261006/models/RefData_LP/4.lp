\ Canonical explicit-bound formulation for WL comparison.
\ Finite bounds and singleton restrictions merged; original variable domains retained.
Maximize
  obj: + 39.619999999999997 x[1] + 35.979999999999997 x[2] + 37.960000000000001 x[3] - 178539 y[1] - 157708 y[2] - 85192 y[3]
Subject To
  min_batch[1]: + 1 x[1] - 18 y[1] >= 0
  min_batch[2]: + 1 x[2] - 25 y[2] >= 0
  min_batch[3]: + 1 x[3] - 23 y[3] >= 0
  linking[1]: + 1 x[1] - 5732 y[1] <= 0
  linking[2]: + 1 x[2] - 5607 y[2] <= 0
  linking[3]: + 1 x[3] - 4653 y[3] <= 0
  shared_production_days: + 0.00170648464163823 x[1] + 0.0030395136778115501 x[2] + 0.0018484288354898299 x[3] <= 22
  __wl_bound_00001_lower: + 1 x[1] >= 0
  __wl_bound_00001_upper: + 1 x[1] <= 5732
  __wl_bound_00002_lower: + 1 x[2] >= 0
  __wl_bound_00002_upper: + 1 x[2] <= 5607
  __wl_bound_00003_lower: + 1 x[3] >= 0
  __wl_bound_00003_upper: + 1 x[3] <= 4653
  __wl_bound_00004_lower: + 1 y[1] >= 0
  __wl_bound_00004_upper: + 1 y[1] <= 1
  __wl_bound_00005_lower: + 1 y[2] >= 0
  __wl_bound_00005_upper: + 1 y[2] <= 1
  __wl_bound_00006_lower: + 1 y[3] >= 0
  __wl_bound_00006_upper: + 1 y[3] <= 1
Bounds
  x[1] free
  x[2] free
  x[3] free
Generals
  x[1] x[2] x[3]
Binaries
  y[1] y[2] y[3]
End
