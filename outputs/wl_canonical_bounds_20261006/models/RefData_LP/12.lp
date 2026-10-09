\ Canonical explicit-bound formulation for WL comparison.
\ Finite bounds and singleton restrictions merged; original variable domains retained.
Minimize
  obj: + 28 x[1,1] + 45 x[1,2] + 60 x[1,3] + 73 x[1,4] + 28 x[2,1] + 45 x[2,2] + 60 x[2,3] + 28 x[3,1] + 45 x[3,2] + 28 x[4,1]
Subject To
  demand_month_1: + 1 x[1,1] + 1 x[1,2] + 1 x[1,3] + 1 x[1,4] >= 1500
  demand_month_2: + 1 x[1,2] + 1 x[1,3] + 1 x[1,4] + 1 x[2,1] + 1 x[2,2] + 1 x[2,3] >= 1000
  demand_month_3: + 1 x[1,3] + 1 x[1,4] + 1 x[2,2] + 1 x[2,3] + 1 x[3,1] + 1 x[3,2] >= 2000
  demand_month_4: + 1 x[1,4] + 1 x[2,3] + 1 x[3,2] + 1 x[4,1] >= 1200
  __wl_bound_00001_lower: + 1 x[1,1] >= 0
  __wl_bound_00002_lower: + 1 x[1,2] >= 0
  __wl_bound_00003_lower: + 1 x[1,3] >= 0
  __wl_bound_00004_lower: + 1 x[1,4] >= 0
  __wl_bound_00005_lower: + 1 x[2,1] >= 0
  __wl_bound_00006_lower: + 1 x[2,2] >= 0
  __wl_bound_00007_lower: + 1 x[2,3] >= 0
  __wl_bound_00008_lower: + 1 x[3,1] >= 0
  __wl_bound_00009_lower: + 1 x[3,2] >= 0
  __wl_bound_00010_lower: + 1 x[4,1] >= 0
Bounds
  x[1,1] free
  x[1,2] free
  x[1,3] free
  x[1,4] free
  x[2,1] free
  x[2,2] free
  x[2,3] free
  x[3,1] free
  x[3,2] free
  x[4,1] free
End
