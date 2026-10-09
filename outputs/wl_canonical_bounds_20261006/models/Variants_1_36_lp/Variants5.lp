\ Canonical explicit-bound formulation for WL comparison.
\ Finite bounds and singleton restrictions merged; original variable domains retained.
Minimize
  obj: + 8 x_M1_J1 + 7 x_M1_J2 + 25 x_M1_J3 + 24 x_M1_J4 + 27 x_M1_J5 + 26 x_M1_J6 + 28 x_M1_J7 + 29 x_M1_J8 + 23 x_M2_J1 + 24 x_M2_J2 + 6 x_M2_J3 + 9 x_M2_J4 + 25 x_M2_J5 + 27 x_M2_J6 + 26 x_M2_J7 + 28 x_M2_J8 + 27 x_M3_J1 + 26 x_M3_J2 + 24 x_M3_J3 + 25 x_M3_J4 + 5 x_M3_J5 + 8 x_M3_J6 + 23 x_M3_J7 + 24 x_M3_J8 + 25 x_M4_J1 + 27 x_M4_J2 + 26 x_M4_J3 + 24 x_M4_J4 + 23 x_M4_J5 + 25 x_M4_J6 + 6 x_M4_J7 + 7 x_M4_J8
Subject To
  assign_J1: + 1 x_M1_J1 + 1 x_M2_J1 + 1 x_M3_J1 + 1 x_M4_J1 = 1
  assign_J2: + 1 x_M1_J2 + 1 x_M2_J2 + 1 x_M3_J2 + 1 x_M4_J2 = 1
  assign_J3: + 1 x_M1_J3 + 1 x_M2_J3 + 1 x_M3_J3 + 1 x_M4_J3 = 1
  assign_J4: + 1 x_M1_J4 + 1 x_M2_J4 + 1 x_M3_J4 + 1 x_M4_J4 = 1
  assign_J5: + 1 x_M1_J5 + 1 x_M2_J5 + 1 x_M3_J5 + 1 x_M4_J5 = 1
  assign_J6: + 1 x_M1_J6 + 1 x_M2_J6 + 1 x_M3_J6 + 1 x_M4_J6 = 1
  assign_J7: + 1 x_M1_J7 + 1 x_M2_J7 + 1 x_M3_J7 + 1 x_M4_J7 = 1
  assign_J8: + 1 x_M1_J8 + 1 x_M2_J8 + 1 x_M3_J8 + 1 x_M4_J8 = 1
  capacity_M1: + 5 x_M1_J1 + 6 x_M1_J2 + 8 x_M1_J3 + 7 x_M1_J4 + 9 x_M1_J5 + 8 x_M1_J6 + 7 x_M1_J7 + 7 x_M1_J8 <= 13
  capacity_M2: + 8 x_M2_J1 + 7 x_M2_J2 + 4 x_M2_J3 + 7 x_M2_J4 + 8 x_M2_J5 + 9 x_M2_J6 + 8 x_M2_J7 + 7 x_M2_J8 <= 12
  capacity_M3: + 9 x_M3_J1 + 8 x_M3_J2 + 7 x_M3_J3 + 8 x_M3_J4 + 6 x_M3_J5 + 5 x_M3_J6 + 8 x_M3_J7 + 7 x_M3_J8 <= 12
  capacity_M4: + 8 x_M4_J1 + 8 x_M4_J2 + 7 x_M4_J3 + 8 x_M4_J4 + 8 x_M4_J5 + 7 x_M4_J6 + 5 x_M4_J7 + 6 x_M4_J8 <= 12
  __wl_bound_00001_lower: + 1 x_M1_J1 >= 0
  __wl_bound_00001_upper: + 1 x_M1_J1 <= 1
  __wl_bound_00002_lower: + 1 x_M1_J2 >= 0
  __wl_bound_00002_upper: + 1 x_M1_J2 <= 1
  __wl_bound_00003_lower: + 1 x_M1_J3 >= 0
  __wl_bound_00003_upper: + 1 x_M1_J3 <= 1
  __wl_bound_00004_lower: + 1 x_M1_J4 >= 0
  __wl_bound_00004_upper: + 1 x_M1_J4 <= 1
  __wl_bound_00005_lower: + 1 x_M1_J5 >= 0
  __wl_bound_00005_upper: + 1 x_M1_J5 <= 1
  __wl_bound_00006_lower: + 1 x_M1_J6 >= 0
  __wl_bound_00006_upper: + 1 x_M1_J6 <= 1
  __wl_bound_00007_lower: + 1 x_M1_J7 >= 0
  __wl_bound_00007_upper: + 1 x_M1_J7 <= 1
  __wl_bound_00008_lower: + 1 x_M1_J8 >= 0
  __wl_bound_00008_upper: + 1 x_M1_J8 <= 1
  __wl_bound_00009_lower: + 1 x_M2_J1 >= 0
  __wl_bound_00009_upper: + 1 x_M2_J1 <= 1
  __wl_bound_00010_lower: + 1 x_M2_J2 >= 0
  __wl_bound_00010_upper: + 1 x_M2_J2 <= 1
  __wl_bound_00011_lower: + 1 x_M2_J3 >= 0
  __wl_bound_00011_upper: + 1 x_M2_J3 <= 1
  __wl_bound_00012_lower: + 1 x_M2_J4 >= 0
  __wl_bound_00012_upper: + 1 x_M2_J4 <= 1
  __wl_bound_00013_lower: + 1 x_M2_J5 >= 0
  __wl_bound_00013_upper: + 1 x_M2_J5 <= 1
  __wl_bound_00014_lower: + 1 x_M2_J6 >= 0
  __wl_bound_00014_upper: + 1 x_M2_J6 <= 1
  __wl_bound_00015_lower: + 1 x_M2_J7 >= 0
  __wl_bound_00015_upper: + 1 x_M2_J7 <= 1
  __wl_bound_00016_lower: + 1 x_M2_J8 >= 0
  __wl_bound_00016_upper: + 1 x_M2_J8 <= 1
  __wl_bound_00017_lower: + 1 x_M3_J1 >= 0
  __wl_bound_00017_upper: + 1 x_M3_J1 <= 1
  __wl_bound_00018_lower: + 1 x_M3_J2 >= 0
  __wl_bound_00018_upper: + 1 x_M3_J2 <= 1
  __wl_bound_00019_lower: + 1 x_M3_J3 >= 0
  __wl_bound_00019_upper: + 1 x_M3_J3 <= 1
  __wl_bound_00020_lower: + 1 x_M3_J4 >= 0
  __wl_bound_00020_upper: + 1 x_M3_J4 <= 1
  __wl_bound_00021_lower: + 1 x_M3_J5 >= 0
  __wl_bound_00021_upper: + 1 x_M3_J5 <= 1
  __wl_bound_00022_lower: + 1 x_M3_J6 >= 0
  __wl_bound_00022_upper: + 1 x_M3_J6 <= 1
  __wl_bound_00023_lower: + 1 x_M3_J7 >= 0
  __wl_bound_00023_upper: + 1 x_M3_J7 <= 1
  __wl_bound_00024_lower: + 1 x_M3_J8 >= 0
  __wl_bound_00024_upper: + 1 x_M3_J8 <= 1
  __wl_bound_00025_lower: + 1 x_M4_J1 >= 0
  __wl_bound_00025_upper: + 1 x_M4_J1 <= 1
  __wl_bound_00026_lower: + 1 x_M4_J2 >= 0
  __wl_bound_00026_upper: + 1 x_M4_J2 <= 1
  __wl_bound_00027_lower: + 1 x_M4_J3 >= 0
  __wl_bound_00027_upper: + 1 x_M4_J3 <= 1
  __wl_bound_00028_lower: + 1 x_M4_J4 >= 0
  __wl_bound_00028_upper: + 1 x_M4_J4 <= 1
  __wl_bound_00029_lower: + 1 x_M4_J5 >= 0
  __wl_bound_00029_upper: + 1 x_M4_J5 <= 1
  __wl_bound_00030_lower: + 1 x_M4_J6 >= 0
  __wl_bound_00030_upper: + 1 x_M4_J6 <= 1
  __wl_bound_00031_lower: + 1 x_M4_J7 >= 0
  __wl_bound_00031_upper: + 1 x_M4_J7 <= 1
  __wl_bound_00032_lower: + 1 x_M4_J8 >= 0
  __wl_bound_00032_upper: + 1 x_M4_J8 <= 1
Bounds
Binaries
  x_M1_J1 x_M1_J2 x_M1_J3 x_M1_J4 x_M1_J5 x_M1_J6 x_M1_J7 x_M1_J8 x_M2_J1 x_M2_J2 x_M2_J3 x_M2_J4 x_M2_J5 x_M2_J6 x_M2_J7 x_M2_J8 x_M3_J1 x_M3_J2 x_M3_J3 x_M3_J4 x_M3_J5 x_M3_J6 x_M3_J7 x_M3_J8 x_M4_J1 x_M4_J2 x_M4_J3 x_M4_J4 x_M4_J5 x_M4_J6 x_M4_J7 x_M4_J8
End
