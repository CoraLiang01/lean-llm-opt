\ Canonical explicit-bound formulation for WL comparison.
\ Finite bounds and singleton restrictions merged; original variable domains retained.
Minimize
  obj: + 3 x_P1_R1 + 260 y_P1_R1 + 4 x_P1_R2 + 300 y_P1_R2 + 18 x_P1_R3 + 620 y_P1_R3 + 20 x_P1_R4 + 670 y_P1_R4 + 22 x_P1_R5 + 710 y_P1_R5 + 19 x_P1_R6 + 660 y_P1_R6 + 17 x_P2_R1 + 610 y_P2_R1 + 16 x_P2_R2 + 590 y_P2_R2 + 4 x_P2_R3 + 280 y_P2_R3 + 5 x_P2_R4 + 310 y_P2_R4 + 20 x_P2_R5 + 690 y_P2_R5 + 18 x_P2_R6 + 640 y_P2_R6 + 21 x_P3_R1 + 700 y_P3_R1 + 19 x_P3_R2 + 650 y_P3_R2 + 18 x_P3_R3 + 630 y_P3_R3 + 17 x_P3_R4 + 600 y_P3_R4 + 3 x_P3_R5 + 250 y_P3_R5 + 4 x_P3_R6 + 290 y_P3_R6
Subject To
  demand_R1: + 1 x_P1_R1 + 1 x_P2_R1 + 1 x_P3_R1 >= 70
  demand_R2: + 1 x_P1_R2 + 1 x_P2_R2 + 1 x_P3_R2 >= 85
  demand_R3: + 1 x_P1_R3 + 1 x_P2_R3 + 1 x_P3_R3 >= 75
  demand_R4: + 1 x_P1_R4 + 1 x_P2_R4 + 1 x_P3_R4 >= 65
  demand_R5: + 1 x_P1_R5 + 1 x_P2_R5 + 1 x_P3_R5 >= 95
  demand_R6: + 1 x_P1_R6 + 1 x_P2_R6 + 1 x_P3_R6 >= 80
  supply_P1: + 1 x_P1_R1 + 1 x_P1_R2 + 1 x_P1_R3 + 1 x_P1_R4 + 1 x_P1_R5 + 1 x_P1_R6 <= 190
  supply_P2: + 1 x_P2_R1 + 1 x_P2_R2 + 1 x_P2_R3 + 1 x_P2_R4 + 1 x_P2_R5 + 1 x_P2_R6 <= 160
  supply_P3: + 1 x_P3_R1 + 1 x_P3_R2 + 1 x_P3_R3 + 1 x_P3_R4 + 1 x_P3_R5 + 1 x_P3_R6 <= 150
  route_link_P1_R1: + 1 x_P1_R1 - 70 y_P1_R1 <= 0
  route_link_P1_R2: + 1 x_P1_R2 - 85 y_P1_R2 <= 0
  route_link_P1_R3: + 1 x_P1_R3 - 75 y_P1_R3 <= 0
  route_link_P1_R4: + 1 x_P1_R4 - 65 y_P1_R4 <= 0
  route_link_P1_R5: + 1 x_P1_R5 - 95 y_P1_R5 <= 0
  route_link_P1_R6: + 1 x_P1_R6 - 80 y_P1_R6 <= 0
  route_link_P2_R1: + 1 x_P2_R1 - 70 y_P2_R1 <= 0
  route_link_P2_R2: + 1 x_P2_R2 - 85 y_P2_R2 <= 0
  route_link_P2_R3: + 1 x_P2_R3 - 75 y_P2_R3 <= 0
  route_link_P2_R4: + 1 x_P2_R4 - 65 y_P2_R4 <= 0
  route_link_P2_R5: + 1 x_P2_R5 - 95 y_P2_R5 <= 0
  route_link_P2_R6: + 1 x_P2_R6 - 80 y_P2_R6 <= 0
  route_link_P3_R1: + 1 x_P3_R1 - 70 y_P3_R1 <= 0
  route_link_P3_R2: + 1 x_P3_R2 - 85 y_P3_R2 <= 0
  route_link_P3_R3: + 1 x_P3_R3 - 75 y_P3_R3 <= 0
  route_link_P3_R4: + 1 x_P3_R4 - 65 y_P3_R4 <= 0
  route_link_P3_R5: + 1 x_P3_R5 - 95 y_P3_R5 <= 0
  route_link_P3_R6: + 1 x_P3_R6 - 80 y_P3_R6 <= 0
  __wl_bound_00001_lower: + 1 x_P1_R1 >= 0
  __wl_bound_00002_lower: + 1 y_P1_R1 >= 0
  __wl_bound_00002_upper: + 1 y_P1_R1 <= 1
  __wl_bound_00003_lower: + 1 x_P1_R2 >= 0
  __wl_bound_00004_lower: + 1 y_P1_R2 >= 0
  __wl_bound_00004_upper: + 1 y_P1_R2 <= 1
  __wl_bound_00005_lower: + 1 x_P1_R3 >= 0
  __wl_bound_00006_lower: + 1 y_P1_R3 >= 0
  __wl_bound_00006_upper: + 1 y_P1_R3 <= 1
  __wl_bound_00007_lower: + 1 x_P1_R4 >= 0
  __wl_bound_00008_lower: + 1 y_P1_R4 >= 0
  __wl_bound_00008_upper: + 1 y_P1_R4 <= 1
  __wl_bound_00009_lower: + 1 x_P1_R5 >= 0
  __wl_bound_00010_lower: + 1 y_P1_R5 >= 0
  __wl_bound_00010_upper: + 1 y_P1_R5 <= 1
  __wl_bound_00011_lower: + 1 x_P1_R6 >= 0
  __wl_bound_00012_lower: + 1 y_P1_R6 >= 0
  __wl_bound_00012_upper: + 1 y_P1_R6 <= 1
  __wl_bound_00013_lower: + 1 x_P2_R1 >= 0
  __wl_bound_00014_lower: + 1 y_P2_R1 >= 0
  __wl_bound_00014_upper: + 1 y_P2_R1 <= 1
  __wl_bound_00015_lower: + 1 x_P2_R2 >= 0
  __wl_bound_00016_lower: + 1 y_P2_R2 >= 0
  __wl_bound_00016_upper: + 1 y_P2_R2 <= 1
  __wl_bound_00017_lower: + 1 x_P2_R3 >= 0
  __wl_bound_00018_lower: + 1 y_P2_R3 >= 0
  __wl_bound_00018_upper: + 1 y_P2_R3 <= 1
  __wl_bound_00019_lower: + 1 x_P2_R4 >= 0
  __wl_bound_00020_lower: + 1 y_P2_R4 >= 0
  __wl_bound_00020_upper: + 1 y_P2_R4 <= 1
  __wl_bound_00021_lower: + 1 x_P2_R5 >= 0
  __wl_bound_00022_lower: + 1 y_P2_R5 >= 0
  __wl_bound_00022_upper: + 1 y_P2_R5 <= 1
  __wl_bound_00023_lower: + 1 x_P2_R6 >= 0
  __wl_bound_00024_lower: + 1 y_P2_R6 >= 0
  __wl_bound_00024_upper: + 1 y_P2_R6 <= 1
  __wl_bound_00025_lower: + 1 x_P3_R1 >= 0
  __wl_bound_00026_lower: + 1 y_P3_R1 >= 0
  __wl_bound_00026_upper: + 1 y_P3_R1 <= 1
  __wl_bound_00027_lower: + 1 x_P3_R2 >= 0
  __wl_bound_00028_lower: + 1 y_P3_R2 >= 0
  __wl_bound_00028_upper: + 1 y_P3_R2 <= 1
  __wl_bound_00029_lower: + 1 x_P3_R3 >= 0
  __wl_bound_00030_lower: + 1 y_P3_R3 >= 0
  __wl_bound_00030_upper: + 1 y_P3_R3 <= 1
  __wl_bound_00031_lower: + 1 x_P3_R4 >= 0
  __wl_bound_00032_lower: + 1 y_P3_R4 >= 0
  __wl_bound_00032_upper: + 1 y_P3_R4 <= 1
  __wl_bound_00033_lower: + 1 x_P3_R5 >= 0
  __wl_bound_00034_lower: + 1 y_P3_R5 >= 0
  __wl_bound_00034_upper: + 1 y_P3_R5 <= 1
  __wl_bound_00035_lower: + 1 x_P3_R6 >= 0
  __wl_bound_00036_lower: + 1 y_P3_R6 >= 0
  __wl_bound_00036_upper: + 1 y_P3_R6 <= 1
Bounds
  x_P1_R1 free
  x_P1_R2 free
  x_P1_R3 free
  x_P1_R4 free
  x_P1_R5 free
  x_P1_R6 free
  x_P2_R1 free
  x_P2_R2 free
  x_P2_R3 free
  x_P2_R4 free
  x_P2_R5 free
  x_P2_R6 free
  x_P3_R1 free
  x_P3_R2 free
  x_P3_R3 free
  x_P3_R4 free
  x_P3_R5 free
  x_P3_R6 free
Binaries
  y_P1_R1 y_P1_R2 y_P1_R3 y_P1_R4 y_P1_R5 y_P1_R6 y_P2_R1 y_P2_R2 y_P2_R3 y_P2_R4 y_P2_R5 y_P2_R6 y_P3_R1 y_P3_R2 y_P3_R3 y_P3_R4 y_P3_R5 y_P3_R6
End
