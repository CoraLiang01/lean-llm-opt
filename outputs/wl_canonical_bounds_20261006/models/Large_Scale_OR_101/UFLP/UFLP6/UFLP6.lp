\ Canonical explicit-bound formulation for WL comparison.
\ Finite bounds and singleton restrictions merged; original variable domains retained.
Minimize
  obj: + 694.67999999999995 x_0_0 + 17.48 x_0_1 + 20.07 x_0_2 + 199.02000000000001 x_0_3 + 1685.53 x_0_4 + 15.130000000000001 x_1_0 + 1.5 x_1_1 + 1.4299999999999999 x_1_2 + 27.879999999999999 x_1_3 + 90.689999999999998 x_1_4 + 2.3399999999999999 x_2_0 + 349.33999999999997 x_2_1 + 246.59999999999999 x_2_2 + 41.299999999999997 x_2_3 + 78.730000000000004 x_2_4 + 1181.5999999999999 x_3_0 + 1458.53 x_3_1 + 1646.3599999999999 x_3_2 + 1924.55 x_3_3 + 38.93 x_3_4 + 1030.8 x_4_0 + 43.479999999999997 x_4_1 + 932.42999999999995 x_4_2 + 55.390000000000001 x_4_3 + 103.84 x_4_4 + 96.579999999999998 y_0 + 94.060000000000002 y_1 + 94.370000000000005 y_2 + 82.879999999999995 y_3 + 94.959999999999994 y_4
Subject To
  R0: + 1 x_0_0 + 1 x_1_0 + 1 x_2_0 + 1 x_3_0 + 1 x_4_0 = 2397
  R1: + 1 x_0_1 + 1 x_1_1 + 1 x_2_1 + 1 x_3_1 + 1 x_4_1 = 1889
  R2: + 1 x_0_2 + 1 x_1_2 + 1 x_2_2 + 1 x_3_2 + 1 x_4_2 = 2518
  R3: + 1 x_0_3 + 1 x_1_3 + 1 x_2_3 + 1 x_3_3 + 1 x_4_3 = 3218
  R4: + 1 x_0_4 + 1 x_1_4 + 1 x_2_4 + 1 x_3_4 + 1 x_4_4 = 1813
  R5: + 1 x_0_0 + 1 x_0_1 + 1 x_0_2 + 1 x_0_3 + 1 x_0_4 - 11835 y_0 <= 0
  R6: + 1 x_1_0 + 1 x_1_1 + 1 x_1_2 + 1 x_1_3 + 1 x_1_4 - 11835 y_1 <= 0
  R7: + 1 x_2_0 + 1 x_2_1 + 1 x_2_2 + 1 x_2_3 + 1 x_2_4 - 11835 y_2 <= 0
  R8: + 1 x_3_0 + 1 x_3_1 + 1 x_3_2 + 1 x_3_3 + 1 x_3_4 - 11835 y_3 <= 0
  R9: + 1 x_4_0 + 1 x_4_1 + 1 x_4_2 + 1 x_4_3 + 1 x_4_4 - 11835 y_4 <= 0
  __wl_bound_00001_lower: + 1 x_0_0 >= 0
  __wl_bound_00002_lower: + 1 x_0_1 >= 0
  __wl_bound_00003_lower: + 1 x_0_2 >= 0
  __wl_bound_00004_lower: + 1 x_0_3 >= 0
  __wl_bound_00005_lower: + 1 x_0_4 >= 0
  __wl_bound_00006_lower: + 1 x_1_0 >= 0
  __wl_bound_00007_lower: + 1 x_1_1 >= 0
  __wl_bound_00008_lower: + 1 x_1_2 >= 0
  __wl_bound_00009_lower: + 1 x_1_3 >= 0
  __wl_bound_00010_lower: + 1 x_1_4 >= 0
  __wl_bound_00011_lower: + 1 x_2_0 >= 0
  __wl_bound_00012_lower: + 1 x_2_1 >= 0
  __wl_bound_00013_lower: + 1 x_2_2 >= 0
  __wl_bound_00014_lower: + 1 x_2_3 >= 0
  __wl_bound_00015_lower: + 1 x_2_4 >= 0
  __wl_bound_00016_lower: + 1 x_3_0 >= 0
  __wl_bound_00017_lower: + 1 x_3_1 >= 0
  __wl_bound_00018_lower: + 1 x_3_2 >= 0
  __wl_bound_00019_lower: + 1 x_3_3 >= 0
  __wl_bound_00020_lower: + 1 x_3_4 >= 0
  __wl_bound_00021_lower: + 1 x_4_0 >= 0
  __wl_bound_00022_lower: + 1 x_4_1 >= 0
  __wl_bound_00023_lower: + 1 x_4_2 >= 0
  __wl_bound_00024_lower: + 1 x_4_3 >= 0
  __wl_bound_00025_lower: + 1 x_4_4 >= 0
  __wl_bound_00026_lower: + 1 y_0 >= 0
  __wl_bound_00026_upper: + 1 y_0 <= 1
  __wl_bound_00027_lower: + 1 y_1 >= 0
  __wl_bound_00027_upper: + 1 y_1 <= 1
  __wl_bound_00028_lower: + 1 y_2 >= 0
  __wl_bound_00028_upper: + 1 y_2 <= 1
  __wl_bound_00029_lower: + 1 y_3 >= 0
  __wl_bound_00029_upper: + 1 y_3 <= 1
  __wl_bound_00030_lower: + 1 y_4 >= 0
  __wl_bound_00030_upper: + 1 y_4 <= 1
Bounds
  x_0_0 free
  x_0_1 free
  x_0_2 free
  x_0_3 free
  x_0_4 free
  x_1_0 free
  x_1_1 free
  x_1_2 free
  x_1_3 free
  x_1_4 free
  x_2_0 free
  x_2_1 free
  x_2_2 free
  x_2_3 free
  x_2_4 free
  x_3_0 free
  x_3_1 free
  x_3_2 free
  x_3_3 free
  x_3_4 free
  x_4_0 free
  x_4_1 free
  x_4_2 free
  x_4_3 free
  x_4_4 free
Binaries
  y_0 y_1 y_2 y_3 y_4
End
