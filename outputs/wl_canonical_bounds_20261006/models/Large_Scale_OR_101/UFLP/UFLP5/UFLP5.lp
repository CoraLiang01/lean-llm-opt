\ Canonical explicit-bound formulation for WL comparison.
\ Finite bounds and singleton restrictions merged; original variable domains retained.
Minimize
  obj: + 150.74000000000001 x_0_0 + 0.02 x_0_1 + 49.130000000000003 x_0_2 + 2080.1500000000001 x_0_3 + 426.39999999999998 x_0_4 + 233.05000000000001 x_1_0 + 97.730000000000004 x_1_1 + 49.840000000000003 x_1_2 + 1982.3900000000001 x_1_3 + 23.960000000000001 x_1_4 + 55.68 x_2_0 + 935.61000000000001 x_2_1 + 4.0300000000000002 x_2_2 + 73.090000000000003 x_2_3 + 525.32000000000005 x_2_4 + 1483.8199999999999 x_3_0 + 1801.0799999999999 x_3_1 + 112.16 x_3_2 + 816.04999999999995 x_3_3 + 107.01000000000001 x_3_4 + 1119.47 x_4_0 + 884.30999999999995 x_4_1 + 0.080000000000000002 x_4_2 + 1544.95 x_4_3 + 543.66999999999996 x_4_4 + 97.650000000000006 y_0 + 99.760000000000005 y_1 + 100.76000000000001 y_2 + 105.31999999999999 y_3 + 98.879999999999995 y_4
Subject To
  R0: + 1 x_0_0 + 1 x_1_0 + 1 x_2_0 + 1 x_3_0 + 1 x_4_0 = 143
  R1: + 1 x_0_1 + 1 x_1_1 + 1 x_2_1 + 1 x_3_1 + 1 x_4_1 = 6
  R2: + 1 x_0_2 + 1 x_1_2 + 1 x_2_2 + 1 x_3_2 + 1 x_4_2 = 10
  R3: + 1 x_0_3 + 1 x_1_3 + 1 x_2_3 + 1 x_3_3 + 1 x_4_3 = 25
  R4: + 1 x_0_4 + 1 x_1_4 + 1 x_2_4 + 1 x_3_4 + 1 x_4_4 = 3
  R5: + 1 x_0_0 + 1 x_0_1 + 1 x_0_2 + 1 x_0_3 + 1 x_0_4 - 187 y_0 <= 0
  R6: + 1 x_1_0 + 1 x_1_1 + 1 x_1_2 + 1 x_1_3 + 1 x_1_4 - 187 y_1 <= 0
  R7: + 1 x_2_0 + 1 x_2_1 + 1 x_2_2 + 1 x_2_3 + 1 x_2_4 - 187 y_2 <= 0
  R8: + 1 x_3_0 + 1 x_3_1 + 1 x_3_2 + 1 x_3_3 + 1 x_3_4 - 187 y_3 <= 0
  R9: + 1 x_4_0 + 1 x_4_1 + 1 x_4_2 + 1 x_4_3 + 1 x_4_4 - 187 y_4 <= 0
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
