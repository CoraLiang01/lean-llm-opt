\ Canonical explicit-bound formulation for WL comparison.
\ Finite bounds and singleton restrictions merged; original variable domains retained.
Maximize
  obj: + 28 x_F1_O1 + 34 x_F1_O2 + 30 x_F1_O3 + 24 x_F1_O4 + 25 x_F2_O1 + 31 x_F2_O2 + 36 x_F2_O3 + 29 x_F2_O4 + 33 x_F3_O1 + 27 x_F3_O2 + 38 x_F3_O3 + 30 x_F3_O4 + 26 x_F4_O1 + 35 x_F4_O2 + 32 x_F4_O3 + 28 x_F4_O4 + 30 x_F5_O1 + 37 x_F5_O2 + 29 x_F5_O3 + 34 x_F5_O4 + 24 x_F6_O1 + 32 x_F6_O2 + 36 x_F6_O3 + 31 x_F6_O4
Subject To
  choose_F1: + 1 x_F1_O1 + 1 x_F1_O2 + 1 x_F1_O3 + 1 x_F1_O4 = 1
  choose_F2: + 1 x_F2_O1 + 1 x_F2_O2 + 1 x_F2_O3 + 1 x_F2_O4 = 1
  choose_F3: + 1 x_F3_O1 + 1 x_F3_O2 + 1 x_F3_O3 + 1 x_F3_O4 = 1
  choose_F4: + 1 x_F4_O1 + 1 x_F4_O2 + 1 x_F4_O3 + 1 x_F4_O4 = 1
  choose_F5: + 1 x_F5_O1 + 1 x_F5_O2 + 1 x_F5_O3 + 1 x_F5_O4 = 1
  choose_F6: + 1 x_F6_O1 + 1 x_F6_O2 + 1 x_F6_O3 + 1 x_F6_O4 = 1
  weight_limit: + 8 x_F1_O1 + 11 x_F1_O2 + 9 x_F1_O3 + 7 x_F1_O4 + 7 x_F2_O1 + 10 x_F2_O2 + 12 x_F2_O3 + 9 x_F2_O4 + 10 x_F3_O1 + 8 x_F3_O2 + 13 x_F3_O3 + 9 x_F3_O4 + 6 x_F4_O1 + 11 x_F4_O2 + 10 x_F4_O3 + 8 x_F4_O4 + 9 x_F5_O1 + 12 x_F5_O2 + 8 x_F5_O3 + 11 x_F5_O4 + 7 x_F6_O1 + 10 x_F6_O2 + 12 x_F6_O3 + 9 x_F6_O4 <= 55
  budget_limit: + 10 x_F1_O1 + 13 x_F1_O2 + 12 x_F1_O3 + 9 x_F1_O4 + 9 x_F2_O1 + 12 x_F2_O2 + 14 x_F2_O3 + 11 x_F2_O4 + 12 x_F3_O1 + 10 x_F3_O2 + 15 x_F3_O3 + 11 x_F3_O4 + 8 x_F4_O1 + 13 x_F4_O2 + 12 x_F4_O3 + 10 x_F4_O4 + 11 x_F5_O1 + 14 x_F5_O2 + 10 x_F5_O3 + 13 x_F5_O4 + 9 x_F6_O1 + 12 x_F6_O2 + 14 x_F6_O3 + 11 x_F6_O4 <= 70
  __wl_bound_00001_lower: + 1 x_F1_O1 >= 0
  __wl_bound_00001_upper: + 1 x_F1_O1 <= 1
  __wl_bound_00002_lower: + 1 x_F1_O2 >= 0
  __wl_bound_00002_upper: + 1 x_F1_O2 <= 1
  __wl_bound_00003_lower: + 1 x_F1_O3 >= 0
  __wl_bound_00003_upper: + 1 x_F1_O3 <= 1
  __wl_bound_00004_lower: + 1 x_F1_O4 >= 0
  __wl_bound_00004_upper: + 1 x_F1_O4 <= 1
  __wl_bound_00005_lower: + 1 x_F2_O1 >= 0
  __wl_bound_00005_upper: + 1 x_F2_O1 <= 1
  __wl_bound_00006_lower: + 1 x_F2_O2 >= 0
  __wl_bound_00006_upper: + 1 x_F2_O2 <= 1
  __wl_bound_00007_lower: + 1 x_F2_O3 >= 0
  __wl_bound_00007_upper: + 1 x_F2_O3 <= 1
  __wl_bound_00008_lower: + 1 x_F2_O4 >= 0
  __wl_bound_00008_upper: + 1 x_F2_O4 <= 1
  __wl_bound_00009_lower: + 1 x_F3_O1 >= 0
  __wl_bound_00009_upper: + 1 x_F3_O1 <= 1
  __wl_bound_00010_lower: + 1 x_F3_O2 >= 0
  __wl_bound_00010_upper: + 1 x_F3_O2 <= 1
  __wl_bound_00011_lower: + 1 x_F3_O3 >= 0
  __wl_bound_00011_upper: + 1 x_F3_O3 <= 1
  __wl_bound_00012_lower: + 1 x_F3_O4 >= 0
  __wl_bound_00012_upper: + 1 x_F3_O4 <= 1
  __wl_bound_00013_lower: + 1 x_F4_O1 >= 0
  __wl_bound_00013_upper: + 1 x_F4_O1 <= 1
  __wl_bound_00014_lower: + 1 x_F4_O2 >= 0
  __wl_bound_00014_upper: + 1 x_F4_O2 <= 1
  __wl_bound_00015_lower: + 1 x_F4_O3 >= 0
  __wl_bound_00015_upper: + 1 x_F4_O3 <= 1
  __wl_bound_00016_lower: + 1 x_F4_O4 >= 0
  __wl_bound_00016_upper: + 1 x_F4_O4 <= 1
  __wl_bound_00017_lower: + 1 x_F5_O1 >= 0
  __wl_bound_00017_upper: + 1 x_F5_O1 <= 1
  __wl_bound_00018_lower: + 1 x_F5_O2 >= 0
  __wl_bound_00018_upper: + 1 x_F5_O2 <= 1
  __wl_bound_00019_lower: + 1 x_F5_O3 >= 0
  __wl_bound_00019_upper: + 1 x_F5_O3 <= 1
  __wl_bound_00020_lower: + 1 x_F5_O4 >= 0
  __wl_bound_00020_upper: + 1 x_F5_O4 <= 1
  __wl_bound_00021_lower: + 1 x_F6_O1 >= 0
  __wl_bound_00021_upper: + 1 x_F6_O1 <= 1
  __wl_bound_00022_lower: + 1 x_F6_O2 >= 0
  __wl_bound_00022_upper: + 1 x_F6_O2 <= 1
  __wl_bound_00023_lower: + 1 x_F6_O3 >= 0
  __wl_bound_00023_upper: + 1 x_F6_O3 <= 1
  __wl_bound_00024_lower: + 1 x_F6_O4 >= 0
  __wl_bound_00024_upper: + 1 x_F6_O4 <= 1
Bounds
Binaries
  x_F1_O1 x_F1_O2 x_F1_O3 x_F1_O4 x_F2_O1 x_F2_O2 x_F2_O3 x_F2_O4 x_F3_O1 x_F3_O2 x_F3_O3 x_F3_O4 x_F4_O1 x_F4_O2 x_F4_O3 x_F4_O4 x_F5_O1 x_F5_O2 x_F5_O3 x_F5_O4 x_F6_O1 x_F6_O2 x_F6_O3 x_F6_O4
End
