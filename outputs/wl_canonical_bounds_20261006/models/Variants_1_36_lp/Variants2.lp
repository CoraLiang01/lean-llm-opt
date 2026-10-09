\ Canonical explicit-bound formulation for WL comparison.
\ Finite bounds and singleton restrictions merged; original variable domains retained.
Minimize
  obj: + 4 x_S1_C1 + 420 y_S1_C1 + 5 x_S1_C2 + 450 y_S1_C2 + 32 x_S1_C3 + 980 y_S1_C3 + 35 x_S1_C4 + 1050 y_S1_C4 + 34 x_S1_C5 + 1010 y_S1_C5 + 36 x_S1_C6 + 1080 y_S1_C6 + 31 x_S1_C7 + 990 y_S1_C7 + 33 x_S1_C8 + 1040 y_S1_C8 + 37 x_S1_C9 + 1120 y_S1_C9 + 35 x_S1_C10 + 1060 y_S1_C10 + 34 x_S1_C11 + 1030 y_S1_C11 + 36 x_S1_C12 + 1090 y_S1_C12 + 33 x_S2_C1 + 1020 y_S2_C1 + 31 x_S2_C2 + 990 y_S2_C2 + 6 x_S2_C3 + 480 y_S2_C3 + 5 x_S2_C4 + 460 y_S2_C4 + 35 x_S2_C5 + 1070 y_S2_C5 + 34 x_S2_C6 + 1040 y_S2_C6 + 36 x_S2_C7 + 1100 y_S2_C7 + 32 x_S2_C8 + 1010 y_S2_C8 + 34 x_S2_C9 + 1050 y_S2_C9 + 37 x_S2_C10 + 1130 y_S2_C10 + 35 x_S2_C11 + 1080 y_S2_C11 + 33 x_S2_C12 + 1000 y_S2_C12 + 34 x_S3_C1 + 1040 y_S3_C1 + 36 x_S3_C2 + 1090 y_S3_C2 + 33 x_S3_C3 + 1010 y_S3_C3 + 31 x_S3_C4 + 980 y_S3_C4 + 4 x_S3_C5 + 500 y_S3_C5 + 7 x_S3_C6 + 530 y_S3_C6
    + 35 x_S3_C7 + 1060 y_S3_C7 + 37 x_S3_C8 + 1110 y_S3_C8 + 32 x_S3_C9 + 990 y_S3_C9 + 34 x_S3_C10 + 1030 y_S3_C10 + 36 x_S3_C11 + 1100 y_S3_C11 + 35 x_S3_C12 + 1060 y_S3_C12 + 36 x_S4_C1 + 1100 y_S4_C1 + 34 x_S4_C2 + 1040 y_S4_C2 + 35 x_S4_C3 + 1080 y_S4_C3 + 33 x_S4_C4 + 1010 y_S4_C4 + 32 x_S4_C5 + 980 y_S4_C5 + 35 x_S4_C6 + 1050 y_S4_C6 + 5 x_S4_C7 + 410 y_S4_C7 + 6 x_S4_C8 + 440 y_S4_C8 + 37 x_S4_C9 + 1120 y_S4_C9 + 31 x_S4_C10 + 970 y_S4_C10 + 34 x_S4_C11 + 1030 y_S4_C11 + 36 x_S4_C12 + 1090 y_S4_C12 + 35 x_S5_C1 + 1060 y_S5_C1 + 37 x_S5_C2 + 1120 y_S5_C2 + 34 x_S5_C3 + 1040 y_S5_C3 + 36 x_S5_C4 + 1090 y_S5_C4 + 33 x_S5_C5 + 1000 y_S5_C5 + 31 x_S5_C6 + 970 y_S5_C6 + 32 x_S5_C7 + 990 y_S5_C7 + 35 x_S5_C8 + 1060 y_S5_C8 + 4 x_S5_C9 + 470 y_S5_C9 + 5 x_S5_C10 + 430 y_S5_C10 + 37 x_S5_C11 + 1130 y_S5_C11 + 34 x_S5_C12 + 1040 y_S5_C12
    + 37 x_S6_C1 + 1120 y_S6_C1 + 35 x_S6_C2 + 1060 y_S6_C2 + 36 x_S6_C3 + 1100 y_S6_C3 + 34 x_S6_C4 + 1040 y_S6_C4 + 35 x_S6_C5 + 1070 y_S6_C5 + 33 x_S6_C6 + 1010 y_S6_C6 + 34 x_S6_C7 + 1030 y_S6_C7 + 32 x_S6_C8 + 980 y_S6_C8 + 36 x_S6_C9 + 1090 y_S6_C9 + 31 x_S6_C10 + 970 y_S6_C10 + 6 x_S6_C11 + 490 y_S6_C11 + 4 x_S6_C12 + 420 y_S6_C12
Subject To
  demand_C1: + 1 x_S1_C1 + 1 x_S2_C1 + 1 x_S3_C1 + 1 x_S4_C1 + 1 x_S5_C1 + 1 x_S6_C1 >= 80
  demand_C2: + 1 x_S1_C2 + 1 x_S2_C2 + 1 x_S3_C2 + 1 x_S4_C2 + 1 x_S5_C2 + 1 x_S6_C2 >= 110
  demand_C3: + 1 x_S1_C3 + 1 x_S2_C3 + 1 x_S3_C3 + 1 x_S4_C3 + 1 x_S5_C3 + 1 x_S6_C3 >= 70
  demand_C4: + 1 x_S1_C4 + 1 x_S2_C4 + 1 x_S3_C4 + 1 x_S4_C4 + 1 x_S5_C4 + 1 x_S6_C4 >= 90
  demand_C5: + 1 x_S1_C5 + 1 x_S2_C5 + 1 x_S3_C5 + 1 x_S4_C5 + 1 x_S5_C5 + 1 x_S6_C5 >= 120
  demand_C6: + 1 x_S1_C6 + 1 x_S2_C6 + 1 x_S3_C6 + 1 x_S4_C6 + 1 x_S5_C6 + 1 x_S6_C6 >= 95
  demand_C7: + 1 x_S1_C7 + 1 x_S2_C7 + 1 x_S3_C7 + 1 x_S4_C7 + 1 x_S5_C7 + 1 x_S6_C7 >= 60
  demand_C8: + 1 x_S1_C8 + 1 x_S2_C8 + 1 x_S3_C8 + 1 x_S4_C8 + 1 x_S5_C8 + 1 x_S6_C8 >= 85
  demand_C9: + 1 x_S1_C9 + 1 x_S2_C9 + 1 x_S3_C9 + 1 x_S4_C9 + 1 x_S5_C9 + 1 x_S6_C9 >= 100
  demand_C10: + 1 x_S1_C10 + 1 x_S2_C10 + 1 x_S3_C10 + 1 x_S4_C10 + 1 x_S5_C10 + 1 x_S6_C10 >= 75
  demand_C11: + 1 x_S1_C11 + 1 x_S2_C11 + 1 x_S3_C11 + 1 x_S4_C11 + 1 x_S5_C11 + 1 x_S6_C11 >= 90
  demand_C12: + 1 x_S1_C12 + 1 x_S2_C12 + 1 x_S3_C12 + 1 x_S4_C12 + 1 x_S5_C12 + 1 x_S6_C12 >= 65
  supply_S1: + 1 x_S1_C1 + 1 x_S1_C2 + 1 x_S1_C3 + 1 x_S1_C4 + 1 x_S1_C5 + 1 x_S1_C6 + 1 x_S1_C7 + 1 x_S1_C8 + 1 x_S1_C9 + 1 x_S1_C10 + 1 x_S1_C11 + 1 x_S1_C12 <= 210
  supply_S2: + 1 x_S2_C1 + 1 x_S2_C2 + 1 x_S2_C3 + 1 x_S2_C4 + 1 x_S2_C5 + 1 x_S2_C6 + 1 x_S2_C7 + 1 x_S2_C8 + 1 x_S2_C9 + 1 x_S2_C10 + 1 x_S2_C11 + 1 x_S2_C12 <= 180
  supply_S3: + 1 x_S3_C1 + 1 x_S3_C2 + 1 x_S3_C3 + 1 x_S3_C4 + 1 x_S3_C5 + 1 x_S3_C6 + 1 x_S3_C7 + 1 x_S3_C8 + 1 x_S3_C9 + 1 x_S3_C10 + 1 x_S3_C11 + 1 x_S3_C12 <= 230
  supply_S4: + 1 x_S4_C1 + 1 x_S4_C2 + 1 x_S4_C3 + 1 x_S4_C4 + 1 x_S4_C5 + 1 x_S4_C6 + 1 x_S4_C7 + 1 x_S4_C8 + 1 x_S4_C9 + 1 x_S4_C10 + 1 x_S4_C11 + 1 x_S4_C12 <= 160
  supply_S5: + 1 x_S5_C1 + 1 x_S5_C2 + 1 x_S5_C3 + 1 x_S5_C4 + 1 x_S5_C5 + 1 x_S5_C6 + 1 x_S5_C7 + 1 x_S5_C8 + 1 x_S5_C9 + 1 x_S5_C10 + 1 x_S5_C11 + 1 x_S5_C12 <= 200
  supply_S6: + 1 x_S6_C1 + 1 x_S6_C2 + 1 x_S6_C3 + 1 x_S6_C4 + 1 x_S6_C5 + 1 x_S6_C6 + 1 x_S6_C7 + 1 x_S6_C8 + 1 x_S6_C9 + 1 x_S6_C10 + 1 x_S6_C11 + 1 x_S6_C12 <= 170
  route_link_S1_C1: + 1 x_S1_C1 - 80 y_S1_C1 <= 0
  route_link_S1_C2: + 1 x_S1_C2 - 110 y_S1_C2 <= 0
  route_link_S1_C3: + 1 x_S1_C3 - 70 y_S1_C3 <= 0
  route_link_S1_C4: + 1 x_S1_C4 - 90 y_S1_C4 <= 0
  route_link_S1_C5: + 1 x_S1_C5 - 120 y_S1_C5 <= 0
  route_link_S1_C6: + 1 x_S1_C6 - 95 y_S1_C6 <= 0
  route_link_S1_C7: + 1 x_S1_C7 - 60 y_S1_C7 <= 0
  route_link_S1_C8: + 1 x_S1_C8 - 85 y_S1_C8 <= 0
  route_link_S1_C9: + 1 x_S1_C9 - 100 y_S1_C9 <= 0
  route_link_S1_C10: + 1 x_S1_C10 - 75 y_S1_C10 <= 0
  route_link_S1_C11: + 1 x_S1_C11 - 90 y_S1_C11 <= 0
  route_link_S1_C12: + 1 x_S1_C12 - 65 y_S1_C12 <= 0
  route_link_S2_C1: + 1 x_S2_C1 - 80 y_S2_C1 <= 0
  route_link_S2_C2: + 1 x_S2_C2 - 110 y_S2_C2 <= 0
  route_link_S2_C3: + 1 x_S2_C3 - 70 y_S2_C3 <= 0
  route_link_S2_C4: + 1 x_S2_C4 - 90 y_S2_C4 <= 0
  route_link_S2_C5: + 1 x_S2_C5 - 120 y_S2_C5 <= 0
  route_link_S2_C6: + 1 x_S2_C6 - 95 y_S2_C6 <= 0
  route_link_S2_C7: + 1 x_S2_C7 - 60 y_S2_C7 <= 0
  route_link_S2_C8: + 1 x_S2_C8 - 85 y_S2_C8 <= 0
  route_link_S2_C9: + 1 x_S2_C9 - 100 y_S2_C9 <= 0
  route_link_S2_C10: + 1 x_S2_C10 - 75 y_S2_C10 <= 0
  route_link_S2_C11: + 1 x_S2_C11 - 90 y_S2_C11 <= 0
  route_link_S2_C12: + 1 x_S2_C12 - 65 y_S2_C12 <= 0
  route_link_S3_C1: + 1 x_S3_C1 - 80 y_S3_C1 <= 0
  route_link_S3_C2: + 1 x_S3_C2 - 110 y_S3_C2 <= 0
  route_link_S3_C3: + 1 x_S3_C3 - 70 y_S3_C3 <= 0
  route_link_S3_C4: + 1 x_S3_C4 - 90 y_S3_C4 <= 0
  route_link_S3_C5: + 1 x_S3_C5 - 120 y_S3_C5 <= 0
  route_link_S3_C6: + 1 x_S3_C6 - 95 y_S3_C6 <= 0
  route_link_S3_C7: + 1 x_S3_C7 - 60 y_S3_C7 <= 0
  route_link_S3_C8: + 1 x_S3_C8 - 85 y_S3_C8 <= 0
  route_link_S3_C9: + 1 x_S3_C9 - 100 y_S3_C9 <= 0
  route_link_S3_C10: + 1 x_S3_C10 - 75 y_S3_C10 <= 0
  route_link_S3_C11: + 1 x_S3_C11 - 90 y_S3_C11 <= 0
  route_link_S3_C12: + 1 x_S3_C12 - 65 y_S3_C12 <= 0
  route_link_S4_C1: + 1 x_S4_C1 - 80 y_S4_C1 <= 0
  route_link_S4_C2: + 1 x_S4_C2 - 110 y_S4_C2 <= 0
  route_link_S4_C3: + 1 x_S4_C3 - 70 y_S4_C3 <= 0
  route_link_S4_C4: + 1 x_S4_C4 - 90 y_S4_C4 <= 0
  route_link_S4_C5: + 1 x_S4_C5 - 120 y_S4_C5 <= 0
  route_link_S4_C6: + 1 x_S4_C6 - 95 y_S4_C6 <= 0
  route_link_S4_C7: + 1 x_S4_C7 - 60 y_S4_C7 <= 0
  route_link_S4_C8: + 1 x_S4_C8 - 85 y_S4_C8 <= 0
  route_link_S4_C9: + 1 x_S4_C9 - 100 y_S4_C9 <= 0
  route_link_S4_C10: + 1 x_S4_C10 - 75 y_S4_C10 <= 0
  route_link_S4_C11: + 1 x_S4_C11 - 90 y_S4_C11 <= 0
  route_link_S4_C12: + 1 x_S4_C12 - 65 y_S4_C12 <= 0
  route_link_S5_C1: + 1 x_S5_C1 - 80 y_S5_C1 <= 0
  route_link_S5_C2: + 1 x_S5_C2 - 110 y_S5_C2 <= 0
  route_link_S5_C3: + 1 x_S5_C3 - 70 y_S5_C3 <= 0
  route_link_S5_C4: + 1 x_S5_C4 - 90 y_S5_C4 <= 0
  route_link_S5_C5: + 1 x_S5_C5 - 120 y_S5_C5 <= 0
  route_link_S5_C6: + 1 x_S5_C6 - 95 y_S5_C6 <= 0
  route_link_S5_C7: + 1 x_S5_C7 - 60 y_S5_C7 <= 0
  route_link_S5_C8: + 1 x_S5_C8 - 85 y_S5_C8 <= 0
  route_link_S5_C9: + 1 x_S5_C9 - 100 y_S5_C9 <= 0
  route_link_S5_C10: + 1 x_S5_C10 - 75 y_S5_C10 <= 0
  route_link_S5_C11: + 1 x_S5_C11 - 90 y_S5_C11 <= 0
  route_link_S5_C12: + 1 x_S5_C12 - 65 y_S5_C12 <= 0
  route_link_S6_C1: + 1 x_S6_C1 - 80 y_S6_C1 <= 0
  route_link_S6_C2: + 1 x_S6_C2 - 110 y_S6_C2 <= 0
  route_link_S6_C3: + 1 x_S6_C3 - 70 y_S6_C3 <= 0
  route_link_S6_C4: + 1 x_S6_C4 - 90 y_S6_C4 <= 0
  route_link_S6_C5: + 1 x_S6_C5 - 120 y_S6_C5 <= 0
  route_link_S6_C6: + 1 x_S6_C6 - 95 y_S6_C6 <= 0
  route_link_S6_C7: + 1 x_S6_C7 - 60 y_S6_C7 <= 0
  route_link_S6_C8: + 1 x_S6_C8 - 85 y_S6_C8 <= 0
  route_link_S6_C9: + 1 x_S6_C9 - 100 y_S6_C9 <= 0
  route_link_S6_C10: + 1 x_S6_C10 - 75 y_S6_C10 <= 0
  route_link_S6_C11: + 1 x_S6_C11 - 90 y_S6_C11 <= 0
  route_link_S6_C12: + 1 x_S6_C12 - 65 y_S6_C12 <= 0
  __wl_bound_00001_lower: + 1 x_S1_C1 >= 0
  __wl_bound_00002_lower: + 1 y_S1_C1 >= 0
  __wl_bound_00002_upper: + 1 y_S1_C1 <= 1
  __wl_bound_00003_lower: + 1 x_S1_C2 >= 0
  __wl_bound_00004_lower: + 1 y_S1_C2 >= 0
  __wl_bound_00004_upper: + 1 y_S1_C2 <= 1
  __wl_bound_00005_lower: + 1 x_S1_C3 >= 0
  __wl_bound_00006_lower: + 1 y_S1_C3 >= 0
  __wl_bound_00006_upper: + 1 y_S1_C3 <= 1
  __wl_bound_00007_lower: + 1 x_S1_C4 >= 0
  __wl_bound_00008_lower: + 1 y_S1_C4 >= 0
  __wl_bound_00008_upper: + 1 y_S1_C4 <= 1
  __wl_bound_00009_lower: + 1 x_S1_C5 >= 0
  __wl_bound_00010_lower: + 1 y_S1_C5 >= 0
  __wl_bound_00010_upper: + 1 y_S1_C5 <= 1
  __wl_bound_00011_lower: + 1 x_S1_C6 >= 0
  __wl_bound_00012_lower: + 1 y_S1_C6 >= 0
  __wl_bound_00012_upper: + 1 y_S1_C6 <= 1
  __wl_bound_00013_lower: + 1 x_S1_C7 >= 0
  __wl_bound_00014_lower: + 1 y_S1_C7 >= 0
  __wl_bound_00014_upper: + 1 y_S1_C7 <= 1
  __wl_bound_00015_lower: + 1 x_S1_C8 >= 0
  __wl_bound_00016_lower: + 1 y_S1_C8 >= 0
  __wl_bound_00016_upper: + 1 y_S1_C8 <= 1
  __wl_bound_00017_lower: + 1 x_S1_C9 >= 0
  __wl_bound_00018_lower: + 1 y_S1_C9 >= 0
  __wl_bound_00018_upper: + 1 y_S1_C9 <= 1
  __wl_bound_00019_lower: + 1 x_S1_C10 >= 0
  __wl_bound_00020_lower: + 1 y_S1_C10 >= 0
  __wl_bound_00020_upper: + 1 y_S1_C10 <= 1
  __wl_bound_00021_lower: + 1 x_S1_C11 >= 0
  __wl_bound_00022_lower: + 1 y_S1_C11 >= 0
  __wl_bound_00022_upper: + 1 y_S1_C11 <= 1
  __wl_bound_00023_lower: + 1 x_S1_C12 >= 0
  __wl_bound_00024_lower: + 1 y_S1_C12 >= 0
  __wl_bound_00024_upper: + 1 y_S1_C12 <= 1
  __wl_bound_00025_lower: + 1 x_S2_C1 >= 0
  __wl_bound_00026_lower: + 1 y_S2_C1 >= 0
  __wl_bound_00026_upper: + 1 y_S2_C1 <= 1
  __wl_bound_00027_lower: + 1 x_S2_C2 >= 0
  __wl_bound_00028_lower: + 1 y_S2_C2 >= 0
  __wl_bound_00028_upper: + 1 y_S2_C2 <= 1
  __wl_bound_00029_lower: + 1 x_S2_C3 >= 0
  __wl_bound_00030_lower: + 1 y_S2_C3 >= 0
  __wl_bound_00030_upper: + 1 y_S2_C3 <= 1
  __wl_bound_00031_lower: + 1 x_S2_C4 >= 0
  __wl_bound_00032_lower: + 1 y_S2_C4 >= 0
  __wl_bound_00032_upper: + 1 y_S2_C4 <= 1
  __wl_bound_00033_lower: + 1 x_S2_C5 >= 0
  __wl_bound_00034_lower: + 1 y_S2_C5 >= 0
  __wl_bound_00034_upper: + 1 y_S2_C5 <= 1
  __wl_bound_00035_lower: + 1 x_S2_C6 >= 0
  __wl_bound_00036_lower: + 1 y_S2_C6 >= 0
  __wl_bound_00036_upper: + 1 y_S2_C6 <= 1
  __wl_bound_00037_lower: + 1 x_S2_C7 >= 0
  __wl_bound_00038_lower: + 1 y_S2_C7 >= 0
  __wl_bound_00038_upper: + 1 y_S2_C7 <= 1
  __wl_bound_00039_lower: + 1 x_S2_C8 >= 0
  __wl_bound_00040_lower: + 1 y_S2_C8 >= 0
  __wl_bound_00040_upper: + 1 y_S2_C8 <= 1
  __wl_bound_00041_lower: + 1 x_S2_C9 >= 0
  __wl_bound_00042_lower: + 1 y_S2_C9 >= 0
  __wl_bound_00042_upper: + 1 y_S2_C9 <= 1
  __wl_bound_00043_lower: + 1 x_S2_C10 >= 0
  __wl_bound_00044_lower: + 1 y_S2_C10 >= 0
  __wl_bound_00044_upper: + 1 y_S2_C10 <= 1
  __wl_bound_00045_lower: + 1 x_S2_C11 >= 0
  __wl_bound_00046_lower: + 1 y_S2_C11 >= 0
  __wl_bound_00046_upper: + 1 y_S2_C11 <= 1
  __wl_bound_00047_lower: + 1 x_S2_C12 >= 0
  __wl_bound_00048_lower: + 1 y_S2_C12 >= 0
  __wl_bound_00048_upper: + 1 y_S2_C12 <= 1
  __wl_bound_00049_lower: + 1 x_S3_C1 >= 0
  __wl_bound_00050_lower: + 1 y_S3_C1 >= 0
  __wl_bound_00050_upper: + 1 y_S3_C1 <= 1
  __wl_bound_00051_lower: + 1 x_S3_C2 >= 0
  __wl_bound_00052_lower: + 1 y_S3_C2 >= 0
  __wl_bound_00052_upper: + 1 y_S3_C2 <= 1
  __wl_bound_00053_lower: + 1 x_S3_C3 >= 0
  __wl_bound_00054_lower: + 1 y_S3_C3 >= 0
  __wl_bound_00054_upper: + 1 y_S3_C3 <= 1
  __wl_bound_00055_lower: + 1 x_S3_C4 >= 0
  __wl_bound_00056_lower: + 1 y_S3_C4 >= 0
  __wl_bound_00056_upper: + 1 y_S3_C4 <= 1
  __wl_bound_00057_lower: + 1 x_S3_C5 >= 0
  __wl_bound_00058_lower: + 1 y_S3_C5 >= 0
  __wl_bound_00058_upper: + 1 y_S3_C5 <= 1
  __wl_bound_00059_lower: + 1 x_S3_C6 >= 0
  __wl_bound_00060_lower: + 1 y_S3_C6 >= 0
  __wl_bound_00060_upper: + 1 y_S3_C6 <= 1
  __wl_bound_00061_lower: + 1 x_S3_C7 >= 0
  __wl_bound_00062_lower: + 1 y_S3_C7 >= 0
  __wl_bound_00062_upper: + 1 y_S3_C7 <= 1
  __wl_bound_00063_lower: + 1 x_S3_C8 >= 0
  __wl_bound_00064_lower: + 1 y_S3_C8 >= 0
  __wl_bound_00064_upper: + 1 y_S3_C8 <= 1
  __wl_bound_00065_lower: + 1 x_S3_C9 >= 0
  __wl_bound_00066_lower: + 1 y_S3_C9 >= 0
  __wl_bound_00066_upper: + 1 y_S3_C9 <= 1
  __wl_bound_00067_lower: + 1 x_S3_C10 >= 0
  __wl_bound_00068_lower: + 1 y_S3_C10 >= 0
  __wl_bound_00068_upper: + 1 y_S3_C10 <= 1
  __wl_bound_00069_lower: + 1 x_S3_C11 >= 0
  __wl_bound_00070_lower: + 1 y_S3_C11 >= 0
  __wl_bound_00070_upper: + 1 y_S3_C11 <= 1
  __wl_bound_00071_lower: + 1 x_S3_C12 >= 0
  __wl_bound_00072_lower: + 1 y_S3_C12 >= 0
  __wl_bound_00072_upper: + 1 y_S3_C12 <= 1
  __wl_bound_00073_lower: + 1 x_S4_C1 >= 0
  __wl_bound_00074_lower: + 1 y_S4_C1 >= 0
  __wl_bound_00074_upper: + 1 y_S4_C1 <= 1
  __wl_bound_00075_lower: + 1 x_S4_C2 >= 0
  __wl_bound_00076_lower: + 1 y_S4_C2 >= 0
  __wl_bound_00076_upper: + 1 y_S4_C2 <= 1
  __wl_bound_00077_lower: + 1 x_S4_C3 >= 0
  __wl_bound_00078_lower: + 1 y_S4_C3 >= 0
  __wl_bound_00078_upper: + 1 y_S4_C3 <= 1
  __wl_bound_00079_lower: + 1 x_S4_C4 >= 0
  __wl_bound_00080_lower: + 1 y_S4_C4 >= 0
  __wl_bound_00080_upper: + 1 y_S4_C4 <= 1
  __wl_bound_00081_lower: + 1 x_S4_C5 >= 0
  __wl_bound_00082_lower: + 1 y_S4_C5 >= 0
  __wl_bound_00082_upper: + 1 y_S4_C5 <= 1
  __wl_bound_00083_lower: + 1 x_S4_C6 >= 0
  __wl_bound_00084_lower: + 1 y_S4_C6 >= 0
  __wl_bound_00084_upper: + 1 y_S4_C6 <= 1
  __wl_bound_00085_lower: + 1 x_S4_C7 >= 0
  __wl_bound_00086_lower: + 1 y_S4_C7 >= 0
  __wl_bound_00086_upper: + 1 y_S4_C7 <= 1
  __wl_bound_00087_lower: + 1 x_S4_C8 >= 0
  __wl_bound_00088_lower: + 1 y_S4_C8 >= 0
  __wl_bound_00088_upper: + 1 y_S4_C8 <= 1
  __wl_bound_00089_lower: + 1 x_S4_C9 >= 0
  __wl_bound_00090_lower: + 1 y_S4_C9 >= 0
  __wl_bound_00090_upper: + 1 y_S4_C9 <= 1
  __wl_bound_00091_lower: + 1 x_S4_C10 >= 0
  __wl_bound_00092_lower: + 1 y_S4_C10 >= 0
  __wl_bound_00092_upper: + 1 y_S4_C10 <= 1
  __wl_bound_00093_lower: + 1 x_S4_C11 >= 0
  __wl_bound_00094_lower: + 1 y_S4_C11 >= 0
  __wl_bound_00094_upper: + 1 y_S4_C11 <= 1
  __wl_bound_00095_lower: + 1 x_S4_C12 >= 0
  __wl_bound_00096_lower: + 1 y_S4_C12 >= 0
  __wl_bound_00096_upper: + 1 y_S4_C12 <= 1
  __wl_bound_00097_lower: + 1 x_S5_C1 >= 0
  __wl_bound_00098_lower: + 1 y_S5_C1 >= 0
  __wl_bound_00098_upper: + 1 y_S5_C1 <= 1
  __wl_bound_00099_lower: + 1 x_S5_C2 >= 0
  __wl_bound_00100_lower: + 1 y_S5_C2 >= 0
  __wl_bound_00100_upper: + 1 y_S5_C2 <= 1
  __wl_bound_00101_lower: + 1 x_S5_C3 >= 0
  __wl_bound_00102_lower: + 1 y_S5_C3 >= 0
  __wl_bound_00102_upper: + 1 y_S5_C3 <= 1
  __wl_bound_00103_lower: + 1 x_S5_C4 >= 0
  __wl_bound_00104_lower: + 1 y_S5_C4 >= 0
  __wl_bound_00104_upper: + 1 y_S5_C4 <= 1
  __wl_bound_00105_lower: + 1 x_S5_C5 >= 0
  __wl_bound_00106_lower: + 1 y_S5_C5 >= 0
  __wl_bound_00106_upper: + 1 y_S5_C5 <= 1
  __wl_bound_00107_lower: + 1 x_S5_C6 >= 0
  __wl_bound_00108_lower: + 1 y_S5_C6 >= 0
  __wl_bound_00108_upper: + 1 y_S5_C6 <= 1
  __wl_bound_00109_lower: + 1 x_S5_C7 >= 0
  __wl_bound_00110_lower: + 1 y_S5_C7 >= 0
  __wl_bound_00110_upper: + 1 y_S5_C7 <= 1
  __wl_bound_00111_lower: + 1 x_S5_C8 >= 0
  __wl_bound_00112_lower: + 1 y_S5_C8 >= 0
  __wl_bound_00112_upper: + 1 y_S5_C8 <= 1
  __wl_bound_00113_lower: + 1 x_S5_C9 >= 0
  __wl_bound_00114_lower: + 1 y_S5_C9 >= 0
  __wl_bound_00114_upper: + 1 y_S5_C9 <= 1
  __wl_bound_00115_lower: + 1 x_S5_C10 >= 0
  __wl_bound_00116_lower: + 1 y_S5_C10 >= 0
  __wl_bound_00116_upper: + 1 y_S5_C10 <= 1
  __wl_bound_00117_lower: + 1 x_S5_C11 >= 0
  __wl_bound_00118_lower: + 1 y_S5_C11 >= 0
  __wl_bound_00118_upper: + 1 y_S5_C11 <= 1
  __wl_bound_00119_lower: + 1 x_S5_C12 >= 0
  __wl_bound_00120_lower: + 1 y_S5_C12 >= 0
  __wl_bound_00120_upper: + 1 y_S5_C12 <= 1
  __wl_bound_00121_lower: + 1 x_S6_C1 >= 0
  __wl_bound_00122_lower: + 1 y_S6_C1 >= 0
  __wl_bound_00122_upper: + 1 y_S6_C1 <= 1
  __wl_bound_00123_lower: + 1 x_S6_C2 >= 0
  __wl_bound_00124_lower: + 1 y_S6_C2 >= 0
  __wl_bound_00124_upper: + 1 y_S6_C2 <= 1
  __wl_bound_00125_lower: + 1 x_S6_C3 >= 0
  __wl_bound_00126_lower: + 1 y_S6_C3 >= 0
  __wl_bound_00126_upper: + 1 y_S6_C3 <= 1
  __wl_bound_00127_lower: + 1 x_S6_C4 >= 0
  __wl_bound_00128_lower: + 1 y_S6_C4 >= 0
  __wl_bound_00128_upper: + 1 y_S6_C4 <= 1
  __wl_bound_00129_lower: + 1 x_S6_C5 >= 0
  __wl_bound_00130_lower: + 1 y_S6_C5 >= 0
  __wl_bound_00130_upper: + 1 y_S6_C5 <= 1
  __wl_bound_00131_lower: + 1 x_S6_C6 >= 0
  __wl_bound_00132_lower: + 1 y_S6_C6 >= 0
  __wl_bound_00132_upper: + 1 y_S6_C6 <= 1
  __wl_bound_00133_lower: + 1 x_S6_C7 >= 0
  __wl_bound_00134_lower: + 1 y_S6_C7 >= 0
  __wl_bound_00134_upper: + 1 y_S6_C7 <= 1
  __wl_bound_00135_lower: + 1 x_S6_C8 >= 0
  __wl_bound_00136_lower: + 1 y_S6_C8 >= 0
  __wl_bound_00136_upper: + 1 y_S6_C8 <= 1
  __wl_bound_00137_lower: + 1 x_S6_C9 >= 0
  __wl_bound_00138_lower: + 1 y_S6_C9 >= 0
  __wl_bound_00138_upper: + 1 y_S6_C9 <= 1
  __wl_bound_00139_lower: + 1 x_S6_C10 >= 0
  __wl_bound_00140_lower: + 1 y_S6_C10 >= 0
  __wl_bound_00140_upper: + 1 y_S6_C10 <= 1
  __wl_bound_00141_lower: + 1 x_S6_C11 >= 0
  __wl_bound_00142_lower: + 1 y_S6_C11 >= 0
  __wl_bound_00142_upper: + 1 y_S6_C11 <= 1
  __wl_bound_00143_lower: + 1 x_S6_C12 >= 0
  __wl_bound_00144_lower: + 1 y_S6_C12 >= 0
  __wl_bound_00144_upper: + 1 y_S6_C12 <= 1
Bounds
  x_S1_C1 free
  x_S1_C2 free
  x_S1_C3 free
  x_S1_C4 free
  x_S1_C5 free
  x_S1_C6 free
  x_S1_C7 free
  x_S1_C8 free
  x_S1_C9 free
  x_S1_C10 free
  x_S1_C11 free
  x_S1_C12 free
  x_S2_C1 free
  x_S2_C2 free
  x_S2_C3 free
  x_S2_C4 free
  x_S2_C5 free
  x_S2_C6 free
  x_S2_C7 free
  x_S2_C8 free
  x_S2_C9 free
  x_S2_C10 free
  x_S2_C11 free
  x_S2_C12 free
  x_S3_C1 free
  x_S3_C2 free
  x_S3_C3 free
  x_S3_C4 free
  x_S3_C5 free
  x_S3_C6 free
  x_S3_C7 free
  x_S3_C8 free
  x_S3_C9 free
  x_S3_C10 free
  x_S3_C11 free
  x_S3_C12 free
  x_S4_C1 free
  x_S4_C2 free
  x_S4_C3 free
  x_S4_C4 free
  x_S4_C5 free
  x_S4_C6 free
  x_S4_C7 free
  x_S4_C8 free
  x_S4_C9 free
  x_S4_C10 free
  x_S4_C11 free
  x_S4_C12 free
  x_S5_C1 free
  x_S5_C2 free
  x_S5_C3 free
  x_S5_C4 free
  x_S5_C5 free
  x_S5_C6 free
  x_S5_C7 free
  x_S5_C8 free
  x_S5_C9 free
  x_S5_C10 free
  x_S5_C11 free
  x_S5_C12 free
  x_S6_C1 free
  x_S6_C2 free
  x_S6_C3 free
  x_S6_C4 free
  x_S6_C5 free
  x_S6_C6 free
  x_S6_C7 free
  x_S6_C8 free
  x_S6_C9 free
  x_S6_C10 free
  x_S6_C11 free
  x_S6_C12 free
Binaries
  y_S1_C1 y_S1_C2 y_S1_C3 y_S1_C4 y_S1_C5 y_S1_C6 y_S1_C7 y_S1_C8 y_S1_C9 y_S1_C10 y_S1_C11 y_S1_C12 y_S2_C1 y_S2_C2 y_S2_C3 y_S2_C4 y_S2_C5 y_S2_C6 y_S2_C7 y_S2_C8 y_S2_C9 y_S2_C10 y_S2_C11 y_S2_C12 y_S3_C1 y_S3_C2 y_S3_C3 y_S3_C4 y_S3_C5 y_S3_C6 y_S3_C7 y_S3_C8 y_S3_C9 y_S3_C10 y_S3_C11 y_S3_C12 y_S4_C1 y_S4_C2 y_S4_C3 y_S4_C4 y_S4_C5 y_S4_C6 y_S4_C7 y_S4_C8 y_S4_C9 y_S4_C10 y_S4_C11 y_S4_C12 y_S5_C1 y_S5_C2 y_S5_C3 y_S5_C4 y_S5_C5 y_S5_C6 y_S5_C7 y_S5_C8 y_S5_C9 y_S5_C10 y_S5_C11 y_S5_C12 y_S6_C1 y_S6_C2 y_S6_C3 y_S6_C4 y_S6_C5 y_S6_C6 y_S6_C7 y_S6_C8 y_S6_C9 y_S6_C10 y_S6_C11 y_S6_C12
End
