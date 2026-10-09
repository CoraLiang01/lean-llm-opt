\ Canonical explicit-bound formulation for WL comparison.
\ Finite bounds and singleton restrictions merged; original variable domains retained.
Minimize
  obj: + 6 y_V1_V2 + 4 y_V1_V3 + 9 y_V1_V4 + 5 y_V2_V3 + 7 y_V2_V5 + 3 y_V3_V4 + 8 y_V3_V6 + 4 y_V4_V6 + 10 y_V4_V7 + 6 y_V5_V6 + 11 y_V5_V8 + 5 y_V6_V7 + 7 y_V6_V8 + 4 y_V7_V8 + 6 y_V7_V9 + 5 y_V8_V9
Subject To
  edge_count: + 1 y_V1_V2 + 1 y_V1_V3 + 1 y_V1_V4 + 1 y_V2_V3 + 1 y_V2_V5 + 1 y_V3_V4 + 1 y_V3_V6 + 1 y_V4_V6 + 1 y_V4_V7 + 1 y_V5_V6 + 1 y_V5_V8 + 1 y_V6_V7 + 1 y_V6_V8 + 1 y_V7_V8 + 1 y_V7_V9 + 1 y_V8_V9 = 8
  flow_balance_V1: + 1 f_V1_V2 - 1 f_V2_V1 + 1 f_V1_V3 - 1 f_V3_V1 + 1 f_V1_V4 - 1 f_V4_V1 = 8
  flow_balance_V2: + 1 f_V1_V2 - 1 f_V2_V1 - 1 f_V2_V3 + 1 f_V3_V2 - 1 f_V2_V5 + 1 f_V5_V2 = 1
  flow_balance_V3: + 1 f_V1_V3 - 1 f_V3_V1 + 1 f_V2_V3 - 1 f_V3_V2 - 1 f_V3_V4 + 1 f_V4_V3 - 1 f_V3_V6 + 1 f_V6_V3 = 1
  flow_balance_V4: + 1 f_V1_V4 - 1 f_V4_V1 + 1 f_V3_V4 - 1 f_V4_V3 - 1 f_V4_V6 + 1 f_V6_V4 - 1 f_V4_V7 + 1 f_V7_V4 = 1
  flow_balance_V5: + 1 f_V2_V5 - 1 f_V5_V2 - 1 f_V5_V6 + 1 f_V6_V5 - 1 f_V5_V8 + 1 f_V8_V5 = 1
  flow_balance_V6: + 1 f_V3_V6 - 1 f_V6_V3 + 1 f_V4_V6 - 1 f_V6_V4 + 1 f_V5_V6 - 1 f_V6_V5 - 1 f_V6_V7 + 1 f_V7_V6 - 1 f_V6_V8 + 1 f_V8_V6 = 1
  flow_balance_V7: + 1 f_V4_V7 - 1 f_V7_V4 + 1 f_V6_V7 - 1 f_V7_V6 - 1 f_V7_V8 + 1 f_V8_V7 - 1 f_V7_V9 + 1 f_V9_V7 = 1
  flow_balance_V8: + 1 f_V5_V8 - 1 f_V8_V5 + 1 f_V6_V8 - 1 f_V8_V6 + 1 f_V7_V8 - 1 f_V8_V7 - 1 f_V8_V9 + 1 f_V9_V8 = 1
  flow_balance_V9: + 1 f_V7_V9 - 1 f_V9_V7 + 1 f_V8_V9 - 1 f_V9_V8 = 1
  flow_link_V1_V2: - 8 y_V1_V2 + 1 f_V1_V2 <= 0
  flow_link_V2_V1: - 8 y_V1_V2 + 1 f_V2_V1 <= 0
  flow_link_V1_V3: - 8 y_V1_V3 + 1 f_V1_V3 <= 0
  flow_link_V3_V1: - 8 y_V1_V3 + 1 f_V3_V1 <= 0
  flow_link_V1_V4: - 8 y_V1_V4 + 1 f_V1_V4 <= 0
  flow_link_V4_V1: - 8 y_V1_V4 + 1 f_V4_V1 <= 0
  flow_link_V2_V3: - 8 y_V2_V3 + 1 f_V2_V3 <= 0
  flow_link_V3_V2: - 8 y_V2_V3 + 1 f_V3_V2 <= 0
  flow_link_V2_V5: - 8 y_V2_V5 + 1 f_V2_V5 <= 0
  flow_link_V5_V2: - 8 y_V2_V5 + 1 f_V5_V2 <= 0
  flow_link_V3_V4: - 8 y_V3_V4 + 1 f_V3_V4 <= 0
  flow_link_V4_V3: - 8 y_V3_V4 + 1 f_V4_V3 <= 0
  flow_link_V3_V6: - 8 y_V3_V6 + 1 f_V3_V6 <= 0
  flow_link_V6_V3: - 8 y_V3_V6 + 1 f_V6_V3 <= 0
  flow_link_V4_V6: - 8 y_V4_V6 + 1 f_V4_V6 <= 0
  flow_link_V6_V4: - 8 y_V4_V6 + 1 f_V6_V4 <= 0
  flow_link_V4_V7: - 8 y_V4_V7 + 1 f_V4_V7 <= 0
  flow_link_V7_V4: - 8 y_V4_V7 + 1 f_V7_V4 <= 0
  flow_link_V5_V6: - 8 y_V5_V6 + 1 f_V5_V6 <= 0
  flow_link_V6_V5: - 8 y_V5_V6 + 1 f_V6_V5 <= 0
  flow_link_V5_V8: - 8 y_V5_V8 + 1 f_V5_V8 <= 0
  flow_link_V8_V5: - 8 y_V5_V8 + 1 f_V8_V5 <= 0
  flow_link_V6_V7: - 8 y_V6_V7 + 1 f_V6_V7 <= 0
  flow_link_V7_V6: - 8 y_V6_V7 + 1 f_V7_V6 <= 0
  flow_link_V6_V8: - 8 y_V6_V8 + 1 f_V6_V8 <= 0
  flow_link_V8_V6: - 8 y_V6_V8 + 1 f_V8_V6 <= 0
  flow_link_V7_V8: - 8 y_V7_V8 + 1 f_V7_V8 <= 0
  flow_link_V8_V7: - 8 y_V7_V8 + 1 f_V8_V7 <= 0
  flow_link_V7_V9: - 8 y_V7_V9 + 1 f_V7_V9 <= 0
  flow_link_V9_V7: - 8 y_V7_V9 + 1 f_V9_V7 <= 0
  flow_link_V8_V9: - 8 y_V8_V9 + 1 f_V8_V9 <= 0
  flow_link_V9_V8: - 8 y_V8_V9 + 1 f_V9_V8 <= 0
  __wl_bound_00001_lower: + 1 y_V1_V2 >= 0
  __wl_bound_00001_upper: + 1 y_V1_V2 <= 1
  __wl_bound_00002_lower: + 1 y_V1_V3 >= 0
  __wl_bound_00002_upper: + 1 y_V1_V3 <= 1
  __wl_bound_00003_lower: + 1 y_V1_V4 >= 0
  __wl_bound_00003_upper: + 1 y_V1_V4 <= 1
  __wl_bound_00004_lower: + 1 y_V2_V3 >= 0
  __wl_bound_00004_upper: + 1 y_V2_V3 <= 1
  __wl_bound_00005_lower: + 1 y_V2_V5 >= 0
  __wl_bound_00005_upper: + 1 y_V2_V5 <= 1
  __wl_bound_00006_lower: + 1 y_V3_V4 >= 0
  __wl_bound_00006_upper: + 1 y_V3_V4 <= 1
  __wl_bound_00007_lower: + 1 y_V3_V6 >= 0
  __wl_bound_00007_upper: + 1 y_V3_V6 <= 1
  __wl_bound_00008_lower: + 1 y_V4_V6 >= 0
  __wl_bound_00008_upper: + 1 y_V4_V6 <= 1
  __wl_bound_00009_lower: + 1 y_V4_V7 >= 0
  __wl_bound_00009_upper: + 1 y_V4_V7 <= 1
  __wl_bound_00010_lower: + 1 y_V5_V6 >= 0
  __wl_bound_00010_upper: + 1 y_V5_V6 <= 1
  __wl_bound_00011_lower: + 1 y_V5_V8 >= 0
  __wl_bound_00011_upper: + 1 y_V5_V8 <= 1
  __wl_bound_00012_lower: + 1 y_V6_V7 >= 0
  __wl_bound_00012_upper: + 1 y_V6_V7 <= 1
  __wl_bound_00013_lower: + 1 y_V6_V8 >= 0
  __wl_bound_00013_upper: + 1 y_V6_V8 <= 1
  __wl_bound_00014_lower: + 1 y_V7_V8 >= 0
  __wl_bound_00014_upper: + 1 y_V7_V8 <= 1
  __wl_bound_00015_lower: + 1 y_V7_V9 >= 0
  __wl_bound_00015_upper: + 1 y_V7_V9 <= 1
  __wl_bound_00016_lower: + 1 y_V8_V9 >= 0
  __wl_bound_00016_upper: + 1 y_V8_V9 <= 1
  __wl_bound_00017_lower: + 1 f_V1_V2 >= 0
  __wl_bound_00018_lower: + 1 f_V2_V1 >= 0
  __wl_bound_00019_lower: + 1 f_V1_V3 >= 0
  __wl_bound_00020_lower: + 1 f_V3_V1 >= 0
  __wl_bound_00021_lower: + 1 f_V1_V4 >= 0
  __wl_bound_00022_lower: + 1 f_V4_V1 >= 0
  __wl_bound_00023_lower: + 1 f_V2_V3 >= 0
  __wl_bound_00024_lower: + 1 f_V3_V2 >= 0
  __wl_bound_00025_lower: + 1 f_V2_V5 >= 0
  __wl_bound_00026_lower: + 1 f_V5_V2 >= 0
  __wl_bound_00027_lower: + 1 f_V3_V4 >= 0
  __wl_bound_00028_lower: + 1 f_V4_V3 >= 0
  __wl_bound_00029_lower: + 1 f_V3_V6 >= 0
  __wl_bound_00030_lower: + 1 f_V6_V3 >= 0
  __wl_bound_00031_lower: + 1 f_V4_V6 >= 0
  __wl_bound_00032_lower: + 1 f_V6_V4 >= 0
  __wl_bound_00033_lower: + 1 f_V4_V7 >= 0
  __wl_bound_00034_lower: + 1 f_V7_V4 >= 0
  __wl_bound_00035_lower: + 1 f_V5_V6 >= 0
  __wl_bound_00036_lower: + 1 f_V6_V5 >= 0
  __wl_bound_00037_lower: + 1 f_V5_V8 >= 0
  __wl_bound_00038_lower: + 1 f_V8_V5 >= 0
  __wl_bound_00039_lower: + 1 f_V6_V7 >= 0
  __wl_bound_00040_lower: + 1 f_V7_V6 >= 0
  __wl_bound_00041_lower: + 1 f_V6_V8 >= 0
  __wl_bound_00042_lower: + 1 f_V8_V6 >= 0
  __wl_bound_00043_lower: + 1 f_V7_V8 >= 0
  __wl_bound_00044_lower: + 1 f_V8_V7 >= 0
  __wl_bound_00045_lower: + 1 f_V7_V9 >= 0
  __wl_bound_00046_lower: + 1 f_V9_V7 >= 0
  __wl_bound_00047_lower: + 1 f_V8_V9 >= 0
  __wl_bound_00048_lower: + 1 f_V9_V8 >= 0
Bounds
  f_V1_V2 free
  f_V2_V1 free
  f_V1_V3 free
  f_V3_V1 free
  f_V1_V4 free
  f_V4_V1 free
  f_V2_V3 free
  f_V3_V2 free
  f_V2_V5 free
  f_V5_V2 free
  f_V3_V4 free
  f_V4_V3 free
  f_V3_V6 free
  f_V6_V3 free
  f_V4_V6 free
  f_V6_V4 free
  f_V4_V7 free
  f_V7_V4 free
  f_V5_V6 free
  f_V6_V5 free
  f_V5_V8 free
  f_V8_V5 free
  f_V6_V7 free
  f_V7_V6 free
  f_V6_V8 free
  f_V8_V6 free
  f_V7_V8 free
  f_V8_V7 free
  f_V7_V9 free
  f_V9_V7 free
  f_V8_V9 free
  f_V9_V8 free
Binaries
  y_V1_V2 y_V1_V3 y_V1_V4 y_V2_V3 y_V2_V5 y_V3_V4 y_V3_V6 y_V4_V6 y_V4_V7 y_V5_V6 y_V5_V8 y_V6_V7 y_V6_V8 y_V7_V8 y_V7_V9 y_V8_V9
End
