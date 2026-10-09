\ Canonical explicit-bound formulation for WL comparison.
\ Finite bounds and singleton restrictions merged; original variable domains retained.
Minimize
  obj: + 6 x_W1_J1 + 8 x_W1_J2 + 18 x_W1_J3 + 20 x_W1_J4 + 21 x_W1_J5 + 19 x_W1_J6 + 23 x_W1_J7 + 22 x_W1_J8 + 24 x_W1_J9 + 19 x_W2_J1 + 18 x_W2_J2 + 7 x_W2_J3 + 6 x_W2_J4 + 20 x_W2_J5 + 22 x_W2_J6 + 21 x_W2_J7 + 23 x_W2_J8 + 25 x_W2_J9 + 22 x_W3_J1 + 21 x_W3_J2 + 20 x_W3_J3 + 19 x_W3_J4 + 5 x_W3_J5 + 7 x_W3_J6 + 18 x_W3_J7 + 20 x_W3_J8 + 21 x_W3_J9 + 21 x_W4_J1 + 22 x_W4_J2 + 23 x_W4_J3 + 20 x_W4_J4 + 19 x_W4_J5 + 18 x_W4_J6 + 6 x_W4_J7 + 8 x_W4_J8 + 7 x_W4_J9
Subject To
  assign_J1: + 1 x_W1_J1 + 1 x_W2_J1 + 1 x_W3_J1 + 1 x_W4_J1 = 1
  assign_J2: + 1 x_W1_J2 + 1 x_W2_J2 + 1 x_W3_J2 + 1 x_W4_J2 = 1
  assign_J3: + 1 x_W1_J3 + 1 x_W2_J3 + 1 x_W3_J3 + 1 x_W4_J3 = 1
  assign_J4: + 1 x_W1_J4 + 1 x_W2_J4 + 1 x_W3_J4 + 1 x_W4_J4 = 1
  assign_J5: + 1 x_W1_J5 + 1 x_W2_J5 + 1 x_W3_J5 + 1 x_W4_J5 = 1
  assign_J6: + 1 x_W1_J6 + 1 x_W2_J6 + 1 x_W3_J6 + 1 x_W4_J6 = 1
  assign_J7: + 1 x_W1_J7 + 1 x_W2_J7 + 1 x_W3_J7 + 1 x_W4_J7 = 1
  assign_J8: + 1 x_W1_J8 + 1 x_W2_J8 + 1 x_W3_J8 + 1 x_W4_J8 = 1
  assign_J9: + 1 x_W1_J9 + 1 x_W2_J9 + 1 x_W3_J9 + 1 x_W4_J9 = 1
  capacity_W1: + 4 x_W1_J1 + 5 x_W1_J2 + 7 x_W1_J3 + 8 x_W1_J4 + 8 x_W1_J5 + 7 x_W1_J6 + 8 x_W1_J7 + 9 x_W1_J8 + 8 x_W1_J9 <= 15
  capacity_W2: + 7 x_W2_J1 + 8 x_W2_J2 + 4 x_W2_J3 + 5 x_W2_J4 + 8 x_W2_J5 + 8 x_W2_J6 + 7 x_W2_J7 + 8 x_W2_J8 + 9 x_W2_J9 <= 14
  capacity_W3: + 8 x_W3_J1 + 7 x_W3_J2 + 8 x_W3_J3 + 7 x_W3_J4 + 5 x_W3_J5 + 4 x_W3_J6 + 7 x_W3_J7 + 8 x_W3_J8 + 7 x_W3_J9 <= 16
  capacity_W4: + 8 x_W4_J1 + 8 x_W4_J2 + 9 x_W4_J3 + 8 x_W4_J4 + 7 x_W4_J5 + 7 x_W4_J6 + 4 x_W4_J7 + 5 x_W4_J8 + 4 x_W4_J9 <= 13
  __wl_bound_00001_lower: + 1 x_W1_J1 >= 0
  __wl_bound_00001_upper: + 1 x_W1_J1 <= 1
  __wl_bound_00002_lower: + 1 x_W1_J2 >= 0
  __wl_bound_00002_upper: + 1 x_W1_J2 <= 1
  __wl_bound_00003_lower: + 1 x_W1_J3 >= 0
  __wl_bound_00003_upper: + 1 x_W1_J3 <= 1
  __wl_bound_00004_lower: + 1 x_W1_J4 >= 0
  __wl_bound_00004_upper: + 1 x_W1_J4 <= 1
  __wl_bound_00005_lower: + 1 x_W1_J5 >= 0
  __wl_bound_00005_upper: + 1 x_W1_J5 <= 1
  __wl_bound_00006_lower: + 1 x_W1_J6 >= 0
  __wl_bound_00006_upper: + 1 x_W1_J6 <= 1
  __wl_bound_00007_lower: + 1 x_W1_J7 >= 0
  __wl_bound_00007_upper: + 1 x_W1_J7 <= 1
  __wl_bound_00008_lower: + 1 x_W1_J8 >= 0
  __wl_bound_00008_upper: + 1 x_W1_J8 <= 1
  __wl_bound_00009_lower: + 1 x_W1_J9 >= 0
  __wl_bound_00009_upper: + 1 x_W1_J9 <= 1
  __wl_bound_00010_lower: + 1 x_W2_J1 >= 0
  __wl_bound_00010_upper: + 1 x_W2_J1 <= 1
  __wl_bound_00011_lower: + 1 x_W2_J2 >= 0
  __wl_bound_00011_upper: + 1 x_W2_J2 <= 1
  __wl_bound_00012_lower: + 1 x_W2_J3 >= 0
  __wl_bound_00012_upper: + 1 x_W2_J3 <= 1
  __wl_bound_00013_lower: + 1 x_W2_J4 >= 0
  __wl_bound_00013_upper: + 1 x_W2_J4 <= 1
  __wl_bound_00014_lower: + 1 x_W2_J5 >= 0
  __wl_bound_00014_upper: + 1 x_W2_J5 <= 1
  __wl_bound_00015_lower: + 1 x_W2_J6 >= 0
  __wl_bound_00015_upper: + 1 x_W2_J6 <= 1
  __wl_bound_00016_lower: + 1 x_W2_J7 >= 0
  __wl_bound_00016_upper: + 1 x_W2_J7 <= 1
  __wl_bound_00017_lower: + 1 x_W2_J8 >= 0
  __wl_bound_00017_upper: + 1 x_W2_J8 <= 1
  __wl_bound_00018_lower: + 1 x_W2_J9 >= 0
  __wl_bound_00018_upper: + 1 x_W2_J9 <= 1
  __wl_bound_00019_lower: + 1 x_W3_J1 >= 0
  __wl_bound_00019_upper: + 1 x_W3_J1 <= 1
  __wl_bound_00020_lower: + 1 x_W3_J2 >= 0
  __wl_bound_00020_upper: + 1 x_W3_J2 <= 1
  __wl_bound_00021_lower: + 1 x_W3_J3 >= 0
  __wl_bound_00021_upper: + 1 x_W3_J3 <= 1
  __wl_bound_00022_lower: + 1 x_W3_J4 >= 0
  __wl_bound_00022_upper: + 1 x_W3_J4 <= 1
  __wl_bound_00023_lower: + 1 x_W3_J5 >= 0
  __wl_bound_00023_upper: + 1 x_W3_J5 <= 1
  __wl_bound_00024_lower: + 1 x_W3_J6 >= 0
  __wl_bound_00024_upper: + 1 x_W3_J6 <= 1
  __wl_bound_00025_lower: + 1 x_W3_J7 >= 0
  __wl_bound_00025_upper: + 1 x_W3_J7 <= 1
  __wl_bound_00026_lower: + 1 x_W3_J8 >= 0
  __wl_bound_00026_upper: + 1 x_W3_J8 <= 1
  __wl_bound_00027_lower: + 1 x_W3_J9 >= 0
  __wl_bound_00027_upper: + 1 x_W3_J9 <= 1
  __wl_bound_00028_lower: + 1 x_W4_J1 >= 0
  __wl_bound_00028_upper: + 1 x_W4_J1 <= 1
  __wl_bound_00029_lower: + 1 x_W4_J2 >= 0
  __wl_bound_00029_upper: + 1 x_W4_J2 <= 1
  __wl_bound_00030_lower: + 1 x_W4_J3 >= 0
  __wl_bound_00030_upper: + 1 x_W4_J3 <= 1
  __wl_bound_00031_lower: + 1 x_W4_J4 >= 0
  __wl_bound_00031_upper: + 1 x_W4_J4 <= 1
  __wl_bound_00032_lower: + 1 x_W4_J5 >= 0
  __wl_bound_00032_upper: + 1 x_W4_J5 <= 1
  __wl_bound_00033_lower: + 1 x_W4_J6 >= 0
  __wl_bound_00033_upper: + 1 x_W4_J6 <= 1
  __wl_bound_00034_lower: + 1 x_W4_J7 >= 0
  __wl_bound_00034_upper: + 1 x_W4_J7 <= 1
  __wl_bound_00035_lower: + 1 x_W4_J8 >= 0
  __wl_bound_00035_upper: + 1 x_W4_J8 <= 1
  __wl_bound_00036_lower: + 1 x_W4_J9 >= 0
  __wl_bound_00036_upper: + 1 x_W4_J9 <= 1
Bounds
Binaries
  x_W1_J1 x_W1_J2 x_W1_J3 x_W1_J4 x_W1_J5 x_W1_J6 x_W1_J7 x_W1_J8 x_W1_J9 x_W2_J1 x_W2_J2 x_W2_J3 x_W2_J4 x_W2_J5 x_W2_J6 x_W2_J7 x_W2_J8 x_W2_J9 x_W3_J1 x_W3_J2 x_W3_J3 x_W3_J4 x_W3_J5 x_W3_J6 x_W3_J7 x_W3_J8 x_W3_J9 x_W4_J1 x_W4_J2 x_W4_J3 x_W4_J4 x_W4_J5 x_W4_J6 x_W4_J7 x_W4_J8 x_W4_J9
End
