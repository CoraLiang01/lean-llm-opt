\ Canonical explicit-bound formulation for WL comparison.
\ Finite bounds and singleton restrictions merged; original variable domains retained.
Minimize
  obj: + 1506.22 x_0_0 + 70.900000000000006 x_0_1 + 8.4399999999999995 x_0_2 + 260.26999999999998 x_0_3 + 197.47 x_0_4 + 71.709999999999994 x_0_5 + 61.189999999999998 x_0_6 + 1732.6500000000001 x_1_0 + 1780.72 x_1_1 + 567.44000000000005 x_1_2 + 448.68000000000001 x_1_3 + 29 x_1_4 + 1484.9100000000001 x_1_5 + 963.91999999999996 x_1_6 + 115.66 x_2_0 + 100.76000000000001 x_2_1 + 64.680000000000007 x_2_2 + 1324.53 x_2_3 + 64.989999999999995 x_2_4 + 134.88 x_2_5 + 2102.8299999999999 x_2_6 + 1254.78 x_3_0 + 1115.6300000000001 x_3_1 + 52.310000000000002 x_3_2 + 1036.1600000000001 x_3_3 + 892.63 x_3_4 + 1464.04 x_3_5 + 1383.4100000000001 x_3_6 + 42.899999999999999 x_4_0 + 891.00999999999999 x_4_1 + 1013.9400000000001 x_4_2 + 1128.72 x_4_3 + 58.909999999999997 x_4_4 + 42.890000000000001 x_4_5 + 1570.3099999999999 x_4_6 + 0.69999999999999996 x_5_0
    + 139.46000000000001 x_5_1 + 70.030000000000001 x_5_2 + 79.150000000000006 x_5_3 + 1482 x_5_4 + 0.91000000000000003 x_5_5 + 110.45999999999999 x_5_6 + 1732.3 x_6_0 + 1780.4400000000001 x_6_1 + 486.5 x_6_2 + 523.74000000000001 x_6_3 + 522.08000000000004 x_6_4 + 82.480000000000004 x_6_5 + 826.40999999999997 x_6_6 + 102.33 y_0 + 94.920000000000002 y_1 + 91.829999999999998 y_2 + 98.709999999999994 y_3 + 95.730000000000004 y_4 + 99.959999999999994 y_5 + 98.159999999999997 y_6
Subject To
  R0: + 1 x_0_0 + 1 x_1_0 + 1 x_2_0 + 1 x_3_0 + 1 x_4_0 + 1 x_5_0 + 1 x_6_0 = 1083
  R1: + 1 x_0_1 + 1 x_1_1 + 1 x_2_1 + 1 x_3_1 + 1 x_4_1 + 1 x_5_1 + 1 x_6_1 = 776
  R2: + 1 x_0_2 + 1 x_1_2 + 1 x_2_2 + 1 x_3_2 + 1 x_4_2 + 1 x_5_2 + 1 x_6_2 = 16214
  R3: + 1 x_0_3 + 1 x_1_3 + 1 x_2_3 + 1 x_3_3 + 1 x_4_3 + 1 x_5_3 + 1 x_6_3 = 553
  R4: + 1 x_0_4 + 1 x_1_4 + 1 x_2_4 + 1 x_3_4 + 1 x_4_4 + 1 x_5_4 + 1 x_6_4 = 17106
  R5: + 1 x_0_5 + 1 x_1_5 + 1 x_2_5 + 1 x_3_5 + 1 x_4_5 + 1 x_5_5 + 1 x_6_5 = 594
  R6: + 1 x_0_6 + 1 x_1_6 + 1 x_2_6 + 1 x_3_6 + 1 x_4_6 + 1 x_5_6 + 1 x_6_6 = 732
  R7: + 1 x_0_0 + 1 x_0_1 + 1 x_0_2 + 1 x_0_3 + 1 x_0_4 + 1 x_0_5 + 1 x_0_6 - 37058 y_0 <= 0
  R8: + 1 x_1_0 + 1 x_1_1 + 1 x_1_2 + 1 x_1_3 + 1 x_1_4 + 1 x_1_5 + 1 x_1_6 - 37058 y_1 <= 0
  R9: + 1 x_2_0 + 1 x_2_1 + 1 x_2_2 + 1 x_2_3 + 1 x_2_4 + 1 x_2_5 + 1 x_2_6 - 37058 y_2 <= 0
  R10: + 1 x_3_0 + 1 x_3_1 + 1 x_3_2 + 1 x_3_3 + 1 x_3_4 + 1 x_3_5 + 1 x_3_6 - 37058 y_3 <= 0
  R11: + 1 x_4_0 + 1 x_4_1 + 1 x_4_2 + 1 x_4_3 + 1 x_4_4 + 1 x_4_5 + 1 x_4_6 - 37058 y_4 <= 0
  R12: + 1 x_5_0 + 1 x_5_1 + 1 x_5_2 + 1 x_5_3 + 1 x_5_4 + 1 x_5_5 + 1 x_5_6 - 37058 y_5 <= 0
  R13: + 1 x_6_0 + 1 x_6_1 + 1 x_6_2 + 1 x_6_3 + 1 x_6_4 + 1 x_6_5 + 1 x_6_6 - 37058 y_6 <= 0
  __wl_bound_00001_lower: + 1 x_0_0 >= 0
  __wl_bound_00002_lower: + 1 x_0_1 >= 0
  __wl_bound_00003_lower: + 1 x_0_2 >= 0
  __wl_bound_00004_lower: + 1 x_0_3 >= 0
  __wl_bound_00005_lower: + 1 x_0_4 >= 0
  __wl_bound_00006_lower: + 1 x_0_5 >= 0
  __wl_bound_00007_lower: + 1 x_0_6 >= 0
  __wl_bound_00008_lower: + 1 x_1_0 >= 0
  __wl_bound_00009_lower: + 1 x_1_1 >= 0
  __wl_bound_00010_lower: + 1 x_1_2 >= 0
  __wl_bound_00011_lower: + 1 x_1_3 >= 0
  __wl_bound_00012_lower: + 1 x_1_4 >= 0
  __wl_bound_00013_lower: + 1 x_1_5 >= 0
  __wl_bound_00014_lower: + 1 x_1_6 >= 0
  __wl_bound_00015_lower: + 1 x_2_0 >= 0
  __wl_bound_00016_lower: + 1 x_2_1 >= 0
  __wl_bound_00017_lower: + 1 x_2_2 >= 0
  __wl_bound_00018_lower: + 1 x_2_3 >= 0
  __wl_bound_00019_lower: + 1 x_2_4 >= 0
  __wl_bound_00020_lower: + 1 x_2_5 >= 0
  __wl_bound_00021_lower: + 1 x_2_6 >= 0
  __wl_bound_00022_lower: + 1 x_3_0 >= 0
  __wl_bound_00023_lower: + 1 x_3_1 >= 0
  __wl_bound_00024_lower: + 1 x_3_2 >= 0
  __wl_bound_00025_lower: + 1 x_3_3 >= 0
  __wl_bound_00026_lower: + 1 x_3_4 >= 0
  __wl_bound_00027_lower: + 1 x_3_5 >= 0
  __wl_bound_00028_lower: + 1 x_3_6 >= 0
  __wl_bound_00029_lower: + 1 x_4_0 >= 0
  __wl_bound_00030_lower: + 1 x_4_1 >= 0
  __wl_bound_00031_lower: + 1 x_4_2 >= 0
  __wl_bound_00032_lower: + 1 x_4_3 >= 0
  __wl_bound_00033_lower: + 1 x_4_4 >= 0
  __wl_bound_00034_lower: + 1 x_4_5 >= 0
  __wl_bound_00035_lower: + 1 x_4_6 >= 0
  __wl_bound_00036_lower: + 1 x_5_0 >= 0
  __wl_bound_00037_lower: + 1 x_5_1 >= 0
  __wl_bound_00038_lower: + 1 x_5_2 >= 0
  __wl_bound_00039_lower: + 1 x_5_3 >= 0
  __wl_bound_00040_lower: + 1 x_5_4 >= 0
  __wl_bound_00041_lower: + 1 x_5_5 >= 0
  __wl_bound_00042_lower: + 1 x_5_6 >= 0
  __wl_bound_00043_lower: + 1 x_6_0 >= 0
  __wl_bound_00044_lower: + 1 x_6_1 >= 0
  __wl_bound_00045_lower: + 1 x_6_2 >= 0
  __wl_bound_00046_lower: + 1 x_6_3 >= 0
  __wl_bound_00047_lower: + 1 x_6_4 >= 0
  __wl_bound_00048_lower: + 1 x_6_5 >= 0
  __wl_bound_00049_lower: + 1 x_6_6 >= 0
  __wl_bound_00050_lower: + 1 y_0 >= 0
  __wl_bound_00050_upper: + 1 y_0 <= 1
  __wl_bound_00051_lower: + 1 y_1 >= 0
  __wl_bound_00051_upper: + 1 y_1 <= 1
  __wl_bound_00052_lower: + 1 y_2 >= 0
  __wl_bound_00052_upper: + 1 y_2 <= 1
  __wl_bound_00053_lower: + 1 y_3 >= 0
  __wl_bound_00053_upper: + 1 y_3 <= 1
  __wl_bound_00054_lower: + 1 y_4 >= 0
  __wl_bound_00054_upper: + 1 y_4 <= 1
  __wl_bound_00055_lower: + 1 y_5 >= 0
  __wl_bound_00055_upper: + 1 y_5 <= 1
  __wl_bound_00056_lower: + 1 y_6 >= 0
  __wl_bound_00056_upper: + 1 y_6 <= 1
Bounds
  x_0_0 free
  x_0_1 free
  x_0_2 free
  x_0_3 free
  x_0_4 free
  x_0_5 free
  x_0_6 free
  x_1_0 free
  x_1_1 free
  x_1_2 free
  x_1_3 free
  x_1_4 free
  x_1_5 free
  x_1_6 free
  x_2_0 free
  x_2_1 free
  x_2_2 free
  x_2_3 free
  x_2_4 free
  x_2_5 free
  x_2_6 free
  x_3_0 free
  x_3_1 free
  x_3_2 free
  x_3_3 free
  x_3_4 free
  x_3_5 free
  x_3_6 free
  x_4_0 free
  x_4_1 free
  x_4_2 free
  x_4_3 free
  x_4_4 free
  x_4_5 free
  x_4_6 free
  x_5_0 free
  x_5_1 free
  x_5_2 free
  x_5_3 free
  x_5_4 free
  x_5_5 free
  x_5_6 free
  x_6_0 free
  x_6_1 free
  x_6_2 free
  x_6_3 free
  x_6_4 free
  x_6_5 free
  x_6_6 free
Binaries
  y_0 y_1 y_2 y_3 y_4 y_5 y_6
End
