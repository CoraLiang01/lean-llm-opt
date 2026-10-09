\ Canonical explicit-bound formulation for WL comparison.
\ Finite bounds and singleton restrictions merged; original variable domains retained.
Minimize
  obj: + 28 X[0,1] + 41 X[0,2] + 63 X[0,3] + 28 X[1,0] + 27 X[1,2] + 87 X[1,3] + 41 X[2,0] + 27 X[2,1] + 81 X[2,3] + 63 X[3,0] + 87 X[3,1] + 81 X[3,2]
Subject To
  depart_0: + 1 X[0,1] + 1 X[0,2] + 1 X[0,3] = 1
  depart_1: + 1 X[1,0] + 1 X[1,2] + 1 X[1,3] = 1
  depart_2: + 1 X[2,0] + 1 X[2,1] + 1 X[2,3] = 1
  depart_3: + 1 X[3,0] + 1 X[3,1] + 1 X[3,2] = 1
  arrive_0: + 1 X[1,0] + 1 X[2,0] + 1 X[3,0] = 1
  arrive_1: + 1 X[0,1] + 1 X[2,1] + 1 X[3,1] = 1
  arrive_2: + 1 X[0,2] + 1 X[1,2] + 1 X[3,2] = 1
  arrive_3: + 1 X[0,3] + 1 X[1,3] + 1 X[2,3] = 1
  mtz_1_2: + 3 X[1,2] + 1 U[1] - 1 U[2] <= 2
  mtz_1_3: + 3 X[1,3] + 1 U[1] - 1 U[3] <= 2
  mtz_2_1: + 3 X[2,1] - 1 U[1] + 1 U[2] <= 2
  mtz_2_3: + 3 X[2,3] + 1 U[2] - 1 U[3] <= 2
  mtz_3_1: + 3 X[3,1] - 1 U[1] + 1 U[3] <= 2
  mtz_3_2: + 3 X[3,2] - 1 U[2] + 1 U[3] <= 2
  __wl_bound_00001_lower: + 1 X[0,1] >= 0
  __wl_bound_00001_upper: + 1 X[0,1] <= 1
  __wl_bound_00002_lower: + 1 X[0,2] >= 0
  __wl_bound_00002_upper: + 1 X[0,2] <= 1
  __wl_bound_00003_lower: + 1 X[0,3] >= 0
  __wl_bound_00003_upper: + 1 X[0,3] <= 1
  __wl_bound_00004_lower: + 1 X[1,0] >= 0
  __wl_bound_00004_upper: + 1 X[1,0] <= 1
  __wl_bound_00005_lower: + 1 X[1,2] >= 0
  __wl_bound_00005_upper: + 1 X[1,2] <= 1
  __wl_bound_00006_lower: + 1 X[1,3] >= 0
  __wl_bound_00006_upper: + 1 X[1,3] <= 1
  __wl_bound_00007_lower: + 1 X[2,0] >= 0
  __wl_bound_00007_upper: + 1 X[2,0] <= 1
  __wl_bound_00008_lower: + 1 X[2,1] >= 0
  __wl_bound_00008_upper: + 1 X[2,1] <= 1
  __wl_bound_00009_lower: + 1 X[2,3] >= 0
  __wl_bound_00009_upper: + 1 X[2,3] <= 1
  __wl_bound_00010_lower: + 1 X[3,0] >= 0
  __wl_bound_00010_upper: + 1 X[3,0] <= 1
  __wl_bound_00011_lower: + 1 X[3,1] >= 0
  __wl_bound_00011_upper: + 1 X[3,1] <= 1
  __wl_bound_00012_lower: + 1 X[3,2] >= 0
  __wl_bound_00012_upper: + 1 X[3,2] <= 1
  __wl_bound_00013_lower: + 1 U[1] >= 1
  __wl_bound_00013_upper: + 1 U[1] <= 3
  __wl_bound_00014_lower: + 1 U[2] >= 1
  __wl_bound_00014_upper: + 1 U[2] <= 3
  __wl_bound_00015_lower: + 1 U[3] >= 1
  __wl_bound_00015_upper: + 1 U[3] <= 3
Bounds
  U[1] free
  U[2] free
  U[3] free
Binaries
  X[0,1] X[0,2] X[0,3] X[1,0] X[1,2] X[1,3] X[2,0] X[2,1] X[2,3] X[3,0] X[3,1] X[3,2]
End
