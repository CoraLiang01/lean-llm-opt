\ Canonical explicit-bound formulation for WL comparison.
\ Finite bounds and singleton restrictions merged; original variable domains retained.
Maximize
  obj: + 18 x_C1_O1 + 27 x_C1_O2 + 30 x_C1_O3 + 24 x_C2_O1 + 32 x_C2_O2 + 28 x_C2_O3 + 22 x_C3_O1 + 35 x_C3_O2 + 31 x_C3_O3 + 20 x_C4_O1 + 29 x_C4_O2 + 34 x_C4_O3 + 23 x_C5_O1 + 33 x_C5_O2 + 26 x_C5_O3 + 21 x_C6_O1 + 30 x_C6_O2 + 36 x_C6_O3
Subject To
  choose_C1: + 1 x_C1_O1 + 1 x_C1_O2 + 1 x_C1_O3 = 1
  choose_C2: + 1 x_C2_O1 + 1 x_C2_O2 + 1 x_C2_O3 = 1
  choose_C3: + 1 x_C3_O1 + 1 x_C3_O2 + 1 x_C3_O3 = 1
  choose_C4: + 1 x_C4_O1 + 1 x_C4_O2 + 1 x_C4_O3 = 1
  choose_C5: + 1 x_C5_O1 + 1 x_C5_O2 + 1 x_C5_O3 = 1
  choose_C6: + 1 x_C6_O1 + 1 x_C6_O2 + 1 x_C6_O3 = 1
  weight_limit: + 5 x_C1_O1 + 8 x_C1_O2 + 10 x_C1_O3 + 7 x_C2_O1 + 11 x_C2_O2 + 9 x_C2_O3 + 6 x_C3_O1 + 12 x_C3_O2 + 10 x_C3_O3 + 5 x_C4_O1 + 9 x_C4_O2 + 12 x_C4_O3 + 7 x_C5_O1 + 11 x_C5_O2 + 8 x_C5_O3 + 6 x_C6_O1 + 10 x_C6_O2 + 13 x_C6_O3 <= 55
  labor_limit: + 6 x_C1_O1 + 9 x_C1_O2 + 11 x_C1_O3 + 8 x_C2_O1 + 13 x_C2_O2 + 10 x_C2_O3 + 7 x_C3_O1 + 14 x_C3_O2 + 12 x_C3_O3 + 8 x_C4_O1 + 11 x_C4_O2 + 13 x_C4_O3 + 7 x_C5_O1 + 12 x_C5_O2 + 10 x_C5_O3 + 6 x_C6_O1 + 11 x_C6_O2 + 15 x_C6_O3 <= 64
  __wl_bound_00001_lower: + 1 x_C1_O1 >= 0
  __wl_bound_00001_upper: + 1 x_C1_O1 <= 1
  __wl_bound_00002_lower: + 1 x_C1_O2 >= 0
  __wl_bound_00002_upper: + 1 x_C1_O2 <= 1
  __wl_bound_00003_lower: + 1 x_C1_O3 >= 0
  __wl_bound_00003_upper: + 1 x_C1_O3 <= 1
  __wl_bound_00004_lower: + 1 x_C2_O1 >= 0
  __wl_bound_00004_upper: + 1 x_C2_O1 <= 1
  __wl_bound_00005_lower: + 1 x_C2_O2 >= 0
  __wl_bound_00005_upper: + 1 x_C2_O2 <= 1
  __wl_bound_00006_lower: + 1 x_C2_O3 >= 0
  __wl_bound_00006_upper: + 1 x_C2_O3 <= 1
  __wl_bound_00007_lower: + 1 x_C3_O1 >= 0
  __wl_bound_00007_upper: + 1 x_C3_O1 <= 1
  __wl_bound_00008_lower: + 1 x_C3_O2 >= 0
  __wl_bound_00008_upper: + 1 x_C3_O2 <= 1
  __wl_bound_00009_lower: + 1 x_C3_O3 >= 0
  __wl_bound_00009_upper: + 1 x_C3_O3 <= 1
  __wl_bound_00010_lower: + 1 x_C4_O1 >= 0
  __wl_bound_00010_upper: + 1 x_C4_O1 <= 1
  __wl_bound_00011_lower: + 1 x_C4_O2 >= 0
  __wl_bound_00011_upper: + 1 x_C4_O2 <= 1
  __wl_bound_00012_lower: + 1 x_C4_O3 >= 0
  __wl_bound_00012_upper: + 1 x_C4_O3 <= 1
  __wl_bound_00013_lower: + 1 x_C5_O1 >= 0
  __wl_bound_00013_upper: + 1 x_C5_O1 <= 1
  __wl_bound_00014_lower: + 1 x_C5_O2 >= 0
  __wl_bound_00014_upper: + 1 x_C5_O2 <= 1
  __wl_bound_00015_lower: + 1 x_C5_O3 >= 0
  __wl_bound_00015_upper: + 1 x_C5_O3 <= 1
  __wl_bound_00016_lower: + 1 x_C6_O1 >= 0
  __wl_bound_00016_upper: + 1 x_C6_O1 <= 1
  __wl_bound_00017_lower: + 1 x_C6_O2 >= 0
  __wl_bound_00017_upper: + 1 x_C6_O2 <= 1
  __wl_bound_00018_lower: + 1 x_C6_O3 >= 0
  __wl_bound_00018_upper: + 1 x_C6_O3 <= 1
Bounds
Binaries
  x_C1_O1 x_C1_O2 x_C1_O3 x_C2_O1 x_C2_O2 x_C2_O3 x_C3_O1 x_C3_O2 x_C3_O3 x_C4_O1 x_C4_O2 x_C4_O3 x_C5_O1 x_C5_O2 x_C5_O3 x_C6_O1 x_C6_O2 x_C6_O3
End
