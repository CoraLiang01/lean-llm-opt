\ Canonical explicit-bound formulation for WL comparison.
\ Finite bounds and singleton restrictions merged; original variable domains retained.
Maximize
  obj: + 1 N
Subject To
  Hours_A: + 1 x[A,1] + 1 x[A,2] + 1 x[A,3] <= 100
  Hours_B: + 1 x[B,1] + 1 x[B,2] + 1 x[B,3] <= 150
  Hours_C: + 1 x[C,1] + 1 x[C,2] + 1 x[C,3] <= 80
  Hours_D: + 1 x[D,1] + 1 x[D,2] + 1 x[D,3] <= 200
  Requirement_1: - 1 N + 10 x[A,1] + 15 x[B,1] + 20 x[C,1] + 10 x[D,1] >= 0
  Requirement_2: - 1 N + 15 x[A,2] + 10 x[B,2] + 5 x[C,2] + 15 x[D,2] >= 0
  Requirement_3: - 1 N + 5 x[A,3] + 5 x[B,3] + 10 x[C,3] + 20 x[D,3] >= 0
  __wl_bound_00001_lower: + 1 N >= 0
  __wl_bound_00002_lower: + 1 x[A,1] >= 0
  __wl_bound_00003_lower: + 1 x[A,2] >= 0
  __wl_bound_00004_lower: + 1 x[A,3] >= 0
  __wl_bound_00005_lower: + 1 x[B,1] >= 0
  __wl_bound_00006_lower: + 1 x[B,2] >= 0
  __wl_bound_00007_lower: + 1 x[B,3] >= 0
  __wl_bound_00008_lower: + 1 x[C,1] >= 0
  __wl_bound_00009_lower: + 1 x[C,2] >= 0
  __wl_bound_00010_lower: + 1 x[C,3] >= 0
  __wl_bound_00011_lower: + 1 x[D,1] >= 0
  __wl_bound_00012_lower: + 1 x[D,2] >= 0
  __wl_bound_00013_lower: + 1 x[D,3] >= 0
Bounds
  N free
  x[A,1] free
  x[A,2] free
  x[A,3] free
  x[B,1] free
  x[B,2] free
  x[B,3] free
  x[C,1] free
  x[C,2] free
  x[C,3] free
  x[D,1] free
  x[D,2] free
  x[D,3] free
End
