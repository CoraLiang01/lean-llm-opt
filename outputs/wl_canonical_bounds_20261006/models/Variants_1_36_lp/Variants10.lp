\ Canonical explicit-bound formulation for WL comparison.
\ Finite bounds and singleton restrictions merged; original variable domains retained.
Minimize
  obj: + 4 y_N1_N2 + 3 y_N1_N3 + 9 y_N1_N4 + 5 y_N2_N3 + 6 y_N2_N5 + 4 y_N3_N4 + 7 y_N3_N6 + 2 y_N4_N6 + 8 y_N4_N7 + 3 y_N5_N6 + 10 y_N5_N8 + 4 y_N6_N7 + 6 y_N6_N8 + 5 y_N7_N8
Subject To
  edge_count: + 1 y_N1_N2 + 1 y_N1_N3 + 1 y_N1_N4 + 1 y_N2_N3 + 1 y_N2_N5 + 1 y_N3_N4 + 1 y_N3_N6 + 1 y_N4_N6 + 1 y_N4_N7 + 1 y_N5_N6 + 1 y_N5_N8 + 1 y_N6_N7 + 1 y_N6_N8 + 1 y_N7_N8 = 7
  flow_balance_N1: + 1 f_N1_N2 - 1 f_N2_N1 + 1 f_N1_N3 - 1 f_N3_N1 + 1 f_N1_N4 - 1 f_N4_N1 = 7
  flow_balance_N2: + 1 f_N1_N2 - 1 f_N2_N1 - 1 f_N2_N3 + 1 f_N3_N2 - 1 f_N2_N5 + 1 f_N5_N2 = 1
  flow_balance_N3: + 1 f_N1_N3 - 1 f_N3_N1 + 1 f_N2_N3 - 1 f_N3_N2 - 1 f_N3_N4 + 1 f_N4_N3 - 1 f_N3_N6 + 1 f_N6_N3 = 1
  flow_balance_N4: + 1 f_N1_N4 - 1 f_N4_N1 + 1 f_N3_N4 - 1 f_N4_N3 - 1 f_N4_N6 + 1 f_N6_N4 - 1 f_N4_N7 + 1 f_N7_N4 = 1
  flow_balance_N5: + 1 f_N2_N5 - 1 f_N5_N2 - 1 f_N5_N6 + 1 f_N6_N5 - 1 f_N5_N8 + 1 f_N8_N5 = 1
  flow_balance_N6: + 1 f_N3_N6 - 1 f_N6_N3 + 1 f_N4_N6 - 1 f_N6_N4 + 1 f_N5_N6 - 1 f_N6_N5 - 1 f_N6_N7 + 1 f_N7_N6 - 1 f_N6_N8 + 1 f_N8_N6 = 1
  flow_balance_N7: + 1 f_N4_N7 - 1 f_N7_N4 + 1 f_N6_N7 - 1 f_N7_N6 - 1 f_N7_N8 + 1 f_N8_N7 = 1
  flow_balance_N8: + 1 f_N5_N8 - 1 f_N8_N5 + 1 f_N6_N8 - 1 f_N8_N6 + 1 f_N7_N8 - 1 f_N8_N7 = 1
  flow_link_N1_N2: - 7 y_N1_N2 + 1 f_N1_N2 <= 0
  flow_link_N2_N1: - 7 y_N1_N2 + 1 f_N2_N1 <= 0
  flow_link_N1_N3: - 7 y_N1_N3 + 1 f_N1_N3 <= 0
  flow_link_N3_N1: - 7 y_N1_N3 + 1 f_N3_N1 <= 0
  flow_link_N1_N4: - 7 y_N1_N4 + 1 f_N1_N4 <= 0
  flow_link_N4_N1: - 7 y_N1_N4 + 1 f_N4_N1 <= 0
  flow_link_N2_N3: - 7 y_N2_N3 + 1 f_N2_N3 <= 0
  flow_link_N3_N2: - 7 y_N2_N3 + 1 f_N3_N2 <= 0
  flow_link_N2_N5: - 7 y_N2_N5 + 1 f_N2_N5 <= 0
  flow_link_N5_N2: - 7 y_N2_N5 + 1 f_N5_N2 <= 0
  flow_link_N3_N4: - 7 y_N3_N4 + 1 f_N3_N4 <= 0
  flow_link_N4_N3: - 7 y_N3_N4 + 1 f_N4_N3 <= 0
  flow_link_N3_N6: - 7 y_N3_N6 + 1 f_N3_N6 <= 0
  flow_link_N6_N3: - 7 y_N3_N6 + 1 f_N6_N3 <= 0
  flow_link_N4_N6: - 7 y_N4_N6 + 1 f_N4_N6 <= 0
  flow_link_N6_N4: - 7 y_N4_N6 + 1 f_N6_N4 <= 0
  flow_link_N4_N7: - 7 y_N4_N7 + 1 f_N4_N7 <= 0
  flow_link_N7_N4: - 7 y_N4_N7 + 1 f_N7_N4 <= 0
  flow_link_N5_N6: - 7 y_N5_N6 + 1 f_N5_N6 <= 0
  flow_link_N6_N5: - 7 y_N5_N6 + 1 f_N6_N5 <= 0
  flow_link_N5_N8: - 7 y_N5_N8 + 1 f_N5_N8 <= 0
  flow_link_N8_N5: - 7 y_N5_N8 + 1 f_N8_N5 <= 0
  flow_link_N6_N7: - 7 y_N6_N7 + 1 f_N6_N7 <= 0
  flow_link_N7_N6: - 7 y_N6_N7 + 1 f_N7_N6 <= 0
  flow_link_N6_N8: - 7 y_N6_N8 + 1 f_N6_N8 <= 0
  flow_link_N8_N6: - 7 y_N6_N8 + 1 f_N8_N6 <= 0
  flow_link_N7_N8: - 7 y_N7_N8 + 1 f_N7_N8 <= 0
  flow_link_N8_N7: - 7 y_N7_N8 + 1 f_N8_N7 <= 0
  __wl_bound_00001_lower: + 1 y_N1_N2 >= 0
  __wl_bound_00001_upper: + 1 y_N1_N2 <= 1
  __wl_bound_00002_lower: + 1 y_N1_N3 >= 0
  __wl_bound_00002_upper: + 1 y_N1_N3 <= 1
  __wl_bound_00003_lower: + 1 y_N1_N4 >= 0
  __wl_bound_00003_upper: + 1 y_N1_N4 <= 1
  __wl_bound_00004_lower: + 1 y_N2_N3 >= 0
  __wl_bound_00004_upper: + 1 y_N2_N3 <= 1
  __wl_bound_00005_lower: + 1 y_N2_N5 >= 0
  __wl_bound_00005_upper: + 1 y_N2_N5 <= 1
  __wl_bound_00006_lower: + 1 y_N3_N4 >= 0
  __wl_bound_00006_upper: + 1 y_N3_N4 <= 1
  __wl_bound_00007_lower: + 1 y_N3_N6 >= 0
  __wl_bound_00007_upper: + 1 y_N3_N6 <= 1
  __wl_bound_00008_lower: + 1 y_N4_N6 >= 0
  __wl_bound_00008_upper: + 1 y_N4_N6 <= 1
  __wl_bound_00009_lower: + 1 y_N4_N7 >= 0
  __wl_bound_00009_upper: + 1 y_N4_N7 <= 1
  __wl_bound_00010_lower: + 1 y_N5_N6 >= 0
  __wl_bound_00010_upper: + 1 y_N5_N6 <= 1
  __wl_bound_00011_lower: + 1 y_N5_N8 >= 0
  __wl_bound_00011_upper: + 1 y_N5_N8 <= 1
  __wl_bound_00012_lower: + 1 y_N6_N7 >= 0
  __wl_bound_00012_upper: + 1 y_N6_N7 <= 1
  __wl_bound_00013_lower: + 1 y_N6_N8 >= 0
  __wl_bound_00013_upper: + 1 y_N6_N8 <= 1
  __wl_bound_00014_lower: + 1 y_N7_N8 >= 0
  __wl_bound_00014_upper: + 1 y_N7_N8 <= 1
  __wl_bound_00015_lower: + 1 f_N1_N2 >= 0
  __wl_bound_00016_lower: + 1 f_N2_N1 >= 0
  __wl_bound_00017_lower: + 1 f_N1_N3 >= 0
  __wl_bound_00018_lower: + 1 f_N3_N1 >= 0
  __wl_bound_00019_lower: + 1 f_N1_N4 >= 0
  __wl_bound_00020_lower: + 1 f_N4_N1 >= 0
  __wl_bound_00021_lower: + 1 f_N2_N3 >= 0
  __wl_bound_00022_lower: + 1 f_N3_N2 >= 0
  __wl_bound_00023_lower: + 1 f_N2_N5 >= 0
  __wl_bound_00024_lower: + 1 f_N5_N2 >= 0
  __wl_bound_00025_lower: + 1 f_N3_N4 >= 0
  __wl_bound_00026_lower: + 1 f_N4_N3 >= 0
  __wl_bound_00027_lower: + 1 f_N3_N6 >= 0
  __wl_bound_00028_lower: + 1 f_N6_N3 >= 0
  __wl_bound_00029_lower: + 1 f_N4_N6 >= 0
  __wl_bound_00030_lower: + 1 f_N6_N4 >= 0
  __wl_bound_00031_lower: + 1 f_N4_N7 >= 0
  __wl_bound_00032_lower: + 1 f_N7_N4 >= 0
  __wl_bound_00033_lower: + 1 f_N5_N6 >= 0
  __wl_bound_00034_lower: + 1 f_N6_N5 >= 0
  __wl_bound_00035_lower: + 1 f_N5_N8 >= 0
  __wl_bound_00036_lower: + 1 f_N8_N5 >= 0
  __wl_bound_00037_lower: + 1 f_N6_N7 >= 0
  __wl_bound_00038_lower: + 1 f_N7_N6 >= 0
  __wl_bound_00039_lower: + 1 f_N6_N8 >= 0
  __wl_bound_00040_lower: + 1 f_N8_N6 >= 0
  __wl_bound_00041_lower: + 1 f_N7_N8 >= 0
  __wl_bound_00042_lower: + 1 f_N8_N7 >= 0
Bounds
  f_N1_N2 free
  f_N2_N1 free
  f_N1_N3 free
  f_N3_N1 free
  f_N1_N4 free
  f_N4_N1 free
  f_N2_N3 free
  f_N3_N2 free
  f_N2_N5 free
  f_N5_N2 free
  f_N3_N4 free
  f_N4_N3 free
  f_N3_N6 free
  f_N6_N3 free
  f_N4_N6 free
  f_N6_N4 free
  f_N4_N7 free
  f_N7_N4 free
  f_N5_N6 free
  f_N6_N5 free
  f_N5_N8 free
  f_N8_N5 free
  f_N6_N7 free
  f_N7_N6 free
  f_N6_N8 free
  f_N8_N6 free
  f_N7_N8 free
  f_N8_N7 free
Binaries
  y_N1_N2 y_N1_N3 y_N1_N4 y_N2_N3 y_N2_N5 y_N3_N4 y_N3_N6 y_N4_N6 y_N4_N7 y_N5_N6 y_N5_N8 y_N6_N7 y_N6_N8 y_N7_N8
End
