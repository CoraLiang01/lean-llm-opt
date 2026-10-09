\ Canonical explicit-bound formulation for WL comparison.
\ Finite bounds and singleton restrictions merged; original variable domains retained.
Minimize
  obj: + 0.080000000000000002 x_0_0 + 52.329999999999998 x_0_1 + 73.569999999999993 x_0_2 + 1237.3299999999999 x_0_3 + 0.070000000000000007 x_0_4 + 112.16 x_0_5 + 46.020000000000003 x_1_0 + 175.22999999999999 x_1_1 + 2026.8299999999999 x_1_2 + 299.88999999999999 x_1_3 + 966.52999999999997 x_1_4 + 1590.4200000000001 x_1_5 + 1031.74 x_2_0 + 78.129999999999995 x_2_1 + 99.019999999999996 x_2_2 + 277.06999999999999 x_2_3 + 884.45000000000005 x_2_4 + 1800.8599999999999 x_2_5 + 868.75 x_3_0 + 94.200000000000003 x_3_1 + 1776.3399999999999 x_3_2 + 285.48000000000002 x_3_3 + 868.85000000000002 x_3_4 + 86.549999999999997 x_3_5 + 1577 x_4_0 + 760.14999999999998 x_4_1 + 2090.1900000000001 x_4_2 + 43.200000000000003 x_4_3 + 1577.1199999999999 x_4_4 + 1095.1700000000001 x_4_5 + 49.140000000000001 x_5_0 + 4.3300000000000001 x_5_1
    + 2079.5700000000002 x_5_2 + 277.04000000000002 x_5_3 + 1032.01 x_5_4 + 1543.49 x_5_5 + 98.879999999999995 y_0 + 99.730000000000004 y_1 + 94.010000000000005 y_2 + 93.769999999999996 y_3 + 107.59 y_4 + 112.65000000000001 y_5
Subject To
  R0: + 1 x_0_0 + 1 x_1_0 + 1 x_2_0 + 1 x_3_0 + 1 x_4_0 + 1 x_5_0 = 216
  R1: + 1 x_0_1 + 1 x_1_1 + 1 x_2_1 + 1 x_3_1 + 1 x_4_1 + 1 x_5_1 = 216
  R2: + 1 x_0_2 + 1 x_1_2 + 1 x_2_2 + 1 x_3_2 + 1 x_4_2 + 1 x_5_2 = 216
  R3: + 1 x_0_3 + 1 x_1_3 + 1 x_2_3 + 1 x_3_3 + 1 x_4_3 + 1 x_5_3 = 144
  R4: + 1 x_0_4 + 1 x_1_4 + 1 x_2_4 + 1 x_3_4 + 1 x_4_4 + 1 x_5_4 = 144
  R5: + 1 x_0_5 + 1 x_1_5 + 1 x_2_5 + 1 x_3_5 + 1 x_4_5 + 1 x_5_5 = 144
  R6: + 1 x_0_0 + 1 x_0_1 + 1 x_0_2 + 1 x_0_3 + 1 x_0_4 + 1 x_0_5 - 1080 y_0 <= 0
  R7: + 1 x_1_0 + 1 x_1_1 + 1 x_1_2 + 1 x_1_3 + 1 x_1_4 + 1 x_1_5 - 1080 y_1 <= 0
  R8: + 1 x_2_0 + 1 x_2_1 + 1 x_2_2 + 1 x_2_3 + 1 x_2_4 + 1 x_2_5 - 1080 y_2 <= 0
  R9: + 1 x_3_0 + 1 x_3_1 + 1 x_3_2 + 1 x_3_3 + 1 x_3_4 + 1 x_3_5 - 1080 y_3 <= 0
  R10: + 1 x_4_0 + 1 x_4_1 + 1 x_4_2 + 1 x_4_3 + 1 x_4_4 + 1 x_4_5 - 1080 y_4 <= 0
  R11: + 1 x_5_0 + 1 x_5_1 + 1 x_5_2 + 1 x_5_3 + 1 x_5_4 + 1 x_5_5 - 1080 y_5 <= 0
  __wl_bound_00001_lower: + 1 x_0_0 >= 0
  __wl_bound_00002_lower: + 1 x_0_1 >= 0
  __wl_bound_00003_lower: + 1 x_0_2 >= 0
  __wl_bound_00004_lower: + 1 x_0_3 >= 0
  __wl_bound_00005_lower: + 1 x_0_4 >= 0
  __wl_bound_00006_lower: + 1 x_0_5 >= 0
  __wl_bound_00007_lower: + 1 x_1_0 >= 0
  __wl_bound_00008_lower: + 1 x_1_1 >= 0
  __wl_bound_00009_lower: + 1 x_1_2 >= 0
  __wl_bound_00010_lower: + 1 x_1_3 >= 0
  __wl_bound_00011_lower: + 1 x_1_4 >= 0
  __wl_bound_00012_lower: + 1 x_1_5 >= 0
  __wl_bound_00013_lower: + 1 x_2_0 >= 0
  __wl_bound_00014_lower: + 1 x_2_1 >= 0
  __wl_bound_00015_lower: + 1 x_2_2 >= 0
  __wl_bound_00016_lower: + 1 x_2_3 >= 0
  __wl_bound_00017_lower: + 1 x_2_4 >= 0
  __wl_bound_00018_lower: + 1 x_2_5 >= 0
  __wl_bound_00019_lower: + 1 x_3_0 >= 0
  __wl_bound_00020_lower: + 1 x_3_1 >= 0
  __wl_bound_00021_lower: + 1 x_3_2 >= 0
  __wl_bound_00022_lower: + 1 x_3_3 >= 0
  __wl_bound_00023_lower: + 1 x_3_4 >= 0
  __wl_bound_00024_lower: + 1 x_3_5 >= 0
  __wl_bound_00025_lower: + 1 x_4_0 >= 0
  __wl_bound_00026_lower: + 1 x_4_1 >= 0
  __wl_bound_00027_lower: + 1 x_4_2 >= 0
  __wl_bound_00028_lower: + 1 x_4_3 >= 0
  __wl_bound_00029_lower: + 1 x_4_4 >= 0
  __wl_bound_00030_lower: + 1 x_4_5 >= 0
  __wl_bound_00031_lower: + 1 x_5_0 >= 0
  __wl_bound_00032_lower: + 1 x_5_1 >= 0
  __wl_bound_00033_lower: + 1 x_5_2 >= 0
  __wl_bound_00034_lower: + 1 x_5_3 >= 0
  __wl_bound_00035_lower: + 1 x_5_4 >= 0
  __wl_bound_00036_lower: + 1 x_5_5 >= 0
  __wl_bound_00037_lower: + 1 y_0 >= 0
  __wl_bound_00037_upper: + 1 y_0 <= 1
  __wl_bound_00038_lower: + 1 y_1 >= 0
  __wl_bound_00038_upper: + 1 y_1 <= 1
  __wl_bound_00039_lower: + 1 y_2 >= 0
  __wl_bound_00039_upper: + 1 y_2 <= 1
  __wl_bound_00040_lower: + 1 y_3 >= 0
  __wl_bound_00040_upper: + 1 y_3 <= 1
  __wl_bound_00041_lower: + 1 y_4 >= 0
  __wl_bound_00041_upper: + 1 y_4 <= 1
  __wl_bound_00042_lower: + 1 y_5 >= 0
  __wl_bound_00042_upper: + 1 y_5 <= 1
Bounds
  x_0_0 free
  x_0_1 free
  x_0_2 free
  x_0_3 free
  x_0_4 free
  x_0_5 free
  x_1_0 free
  x_1_1 free
  x_1_2 free
  x_1_3 free
  x_1_4 free
  x_1_5 free
  x_2_0 free
  x_2_1 free
  x_2_2 free
  x_2_3 free
  x_2_4 free
  x_2_5 free
  x_3_0 free
  x_3_1 free
  x_3_2 free
  x_3_3 free
  x_3_4 free
  x_3_5 free
  x_4_0 free
  x_4_1 free
  x_4_2 free
  x_4_3 free
  x_4_4 free
  x_4_5 free
  x_5_0 free
  x_5_1 free
  x_5_2 free
  x_5_3 free
  x_5_4 free
  x_5_5 free
Binaries
  y_0 y_1 y_2 y_3 y_4 y_5
End
