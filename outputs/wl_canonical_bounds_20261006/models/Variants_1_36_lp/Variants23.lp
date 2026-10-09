\ Canonical explicit-bound formulation for WL comparison.
\ Finite bounds and singleton restrictions merged; original variable domains retained.
Minimize
  obj: + 4 x_D1_M1 + 240 y_D1_M1 + 5 x_D1_M2 + 270 y_D1_M2 + 17 x_D1_M3 + 560 y_D1_M3 + 18 x_D1_M4 + 590 y_D1_M4 + 16 x_D1_M5 + 540 y_D1_M5 + 15 x_D2_M1 + 520 y_D2_M1 + 14 x_D2_M2 + 500 y_D2_M2 + 3 x_D2_M3 + 230 y_D2_M3 + 6 x_D2_M4 + 280 y_D2_M4 + 17 x_D2_M5 + 570 y_D2_M5 + 18 x_D3_M1 + 610 y_D3_M1 + 16 x_D3_M2 + 580 y_D3_M2 + 15 x_D3_M3 + 540 y_D3_M3 + 4 x_D3_M4 + 260 y_D3_M4 + 5 x_D3_M5 + 250 y_D3_M5 + 6 x_D4_M1 + 290 y_D4_M1 + 7 x_D4_M2 + 320 y_D4_M2 + 14 x_D4_M3 + 500 y_D4_M3 + 16 x_D4_M4 + 550 y_D4_M4 + 13 x_D4_M5 + 480 y_D4_M5
Subject To
  demand_M1: + 1 x_D1_M1 + 1 x_D2_M1 + 1 x_D3_M1 + 1 x_D4_M1 >= 65
  demand_M2: + 1 x_D1_M2 + 1 x_D2_M2 + 1 x_D3_M2 + 1 x_D4_M2 >= 80
  demand_M3: + 1 x_D1_M3 + 1 x_D2_M3 + 1 x_D3_M3 + 1 x_D4_M3 >= 75
  demand_M4: + 1 x_D1_M4 + 1 x_D2_M4 + 1 x_D3_M4 + 1 x_D4_M4 >= 95
  demand_M5: + 1 x_D1_M5 + 1 x_D2_M5 + 1 x_D3_M5 + 1 x_D4_M5 >= 70
  supply_D1: + 1 x_D1_M1 + 1 x_D1_M2 + 1 x_D1_M3 + 1 x_D1_M4 + 1 x_D1_M5 <= 120
  supply_D2: + 1 x_D2_M1 + 1 x_D2_M2 + 1 x_D2_M3 + 1 x_D2_M4 + 1 x_D2_M5 <= 100
  supply_D3: + 1 x_D3_M1 + 1 x_D3_M2 + 1 x_D3_M3 + 1 x_D3_M4 + 1 x_D3_M5 <= 140
  supply_D4: + 1 x_D4_M1 + 1 x_D4_M2 + 1 x_D4_M3 + 1 x_D4_M4 + 1 x_D4_M5 <= 90
  route_link_D1_M1: + 1 x_D1_M1 - 65 y_D1_M1 <= 0
  route_link_D1_M2: + 1 x_D1_M2 - 80 y_D1_M2 <= 0
  route_link_D1_M3: + 1 x_D1_M3 - 75 y_D1_M3 <= 0
  route_link_D1_M4: + 1 x_D1_M4 - 95 y_D1_M4 <= 0
  route_link_D1_M5: + 1 x_D1_M5 - 70 y_D1_M5 <= 0
  route_link_D2_M1: + 1 x_D2_M1 - 65 y_D2_M1 <= 0
  route_link_D2_M2: + 1 x_D2_M2 - 80 y_D2_M2 <= 0
  route_link_D2_M3: + 1 x_D2_M3 - 75 y_D2_M3 <= 0
  route_link_D2_M4: + 1 x_D2_M4 - 95 y_D2_M4 <= 0
  route_link_D2_M5: + 1 x_D2_M5 - 70 y_D2_M5 <= 0
  route_link_D3_M1: + 1 x_D3_M1 - 65 y_D3_M1 <= 0
  route_link_D3_M2: + 1 x_D3_M2 - 80 y_D3_M2 <= 0
  route_link_D3_M3: + 1 x_D3_M3 - 75 y_D3_M3 <= 0
  route_link_D3_M4: + 1 x_D3_M4 - 95 y_D3_M4 <= 0
  route_link_D3_M5: + 1 x_D3_M5 - 70 y_D3_M5 <= 0
  route_link_D4_M1: + 1 x_D4_M1 - 65 y_D4_M1 <= 0
  route_link_D4_M2: + 1 x_D4_M2 - 80 y_D4_M2 <= 0
  route_link_D4_M3: + 1 x_D4_M3 - 75 y_D4_M3 <= 0
  route_link_D4_M4: + 1 x_D4_M4 - 90 y_D4_M4 <= 0
  route_link_D4_M5: + 1 x_D4_M5 - 70 y_D4_M5 <= 0
  __wl_bound_00001_lower: + 1 x_D1_M1 >= 0
  __wl_bound_00002_lower: + 1 y_D1_M1 >= 0
  __wl_bound_00002_upper: + 1 y_D1_M1 <= 1
  __wl_bound_00003_lower: + 1 x_D1_M2 >= 0
  __wl_bound_00004_lower: + 1 y_D1_M2 >= 0
  __wl_bound_00004_upper: + 1 y_D1_M2 <= 1
  __wl_bound_00005_lower: + 1 x_D1_M3 >= 0
  __wl_bound_00006_lower: + 1 y_D1_M3 >= 0
  __wl_bound_00006_upper: + 1 y_D1_M3 <= 1
  __wl_bound_00007_lower: + 1 x_D1_M4 >= 0
  __wl_bound_00008_lower: + 1 y_D1_M4 >= 0
  __wl_bound_00008_upper: + 1 y_D1_M4 <= 1
  __wl_bound_00009_lower: + 1 x_D1_M5 >= 0
  __wl_bound_00010_lower: + 1 y_D1_M5 >= 0
  __wl_bound_00010_upper: + 1 y_D1_M5 <= 1
  __wl_bound_00011_lower: + 1 x_D2_M1 >= 0
  __wl_bound_00012_lower: + 1 y_D2_M1 >= 0
  __wl_bound_00012_upper: + 1 y_D2_M1 <= 1
  __wl_bound_00013_lower: + 1 x_D2_M2 >= 0
  __wl_bound_00014_lower: + 1 y_D2_M2 >= 0
  __wl_bound_00014_upper: + 1 y_D2_M2 <= 1
  __wl_bound_00015_lower: + 1 x_D2_M3 >= 0
  __wl_bound_00016_lower: + 1 y_D2_M3 >= 0
  __wl_bound_00016_upper: + 1 y_D2_M3 <= 1
  __wl_bound_00017_lower: + 1 x_D2_M4 >= 0
  __wl_bound_00018_lower: + 1 y_D2_M4 >= 0
  __wl_bound_00018_upper: + 1 y_D2_M4 <= 1
  __wl_bound_00019_lower: + 1 x_D2_M5 >= 0
  __wl_bound_00020_lower: + 1 y_D2_M5 >= 0
  __wl_bound_00020_upper: + 1 y_D2_M5 <= 1
  __wl_bound_00021_lower: + 1 x_D3_M1 >= 0
  __wl_bound_00022_lower: + 1 y_D3_M1 >= 0
  __wl_bound_00022_upper: + 1 y_D3_M1 <= 1
  __wl_bound_00023_lower: + 1 x_D3_M2 >= 0
  __wl_bound_00024_lower: + 1 y_D3_M2 >= 0
  __wl_bound_00024_upper: + 1 y_D3_M2 <= 1
  __wl_bound_00025_lower: + 1 x_D3_M3 >= 0
  __wl_bound_00026_lower: + 1 y_D3_M3 >= 0
  __wl_bound_00026_upper: + 1 y_D3_M3 <= 1
  __wl_bound_00027_lower: + 1 x_D3_M4 >= 0
  __wl_bound_00028_lower: + 1 y_D3_M4 >= 0
  __wl_bound_00028_upper: + 1 y_D3_M4 <= 1
  __wl_bound_00029_lower: + 1 x_D3_M5 >= 0
  __wl_bound_00030_lower: + 1 y_D3_M5 >= 0
  __wl_bound_00030_upper: + 1 y_D3_M5 <= 1
  __wl_bound_00031_lower: + 1 x_D4_M1 >= 0
  __wl_bound_00032_lower: + 1 y_D4_M1 >= 0
  __wl_bound_00032_upper: + 1 y_D4_M1 <= 1
  __wl_bound_00033_lower: + 1 x_D4_M2 >= 0
  __wl_bound_00034_lower: + 1 y_D4_M2 >= 0
  __wl_bound_00034_upper: + 1 y_D4_M2 <= 1
  __wl_bound_00035_lower: + 1 x_D4_M3 >= 0
  __wl_bound_00036_lower: + 1 y_D4_M3 >= 0
  __wl_bound_00036_upper: + 1 y_D4_M3 <= 1
  __wl_bound_00037_lower: + 1 x_D4_M4 >= 0
  __wl_bound_00038_lower: + 1 y_D4_M4 >= 0
  __wl_bound_00038_upper: + 1 y_D4_M4 <= 1
  __wl_bound_00039_lower: + 1 x_D4_M5 >= 0
  __wl_bound_00040_lower: + 1 y_D4_M5 >= 0
  __wl_bound_00040_upper: + 1 y_D4_M5 <= 1
Bounds
  x_D1_M1 free
  x_D1_M2 free
  x_D1_M3 free
  x_D1_M4 free
  x_D1_M5 free
  x_D2_M1 free
  x_D2_M2 free
  x_D2_M3 free
  x_D2_M4 free
  x_D2_M5 free
  x_D3_M1 free
  x_D3_M2 free
  x_D3_M3 free
  x_D3_M4 free
  x_D3_M5 free
  x_D4_M1 free
  x_D4_M2 free
  x_D4_M3 free
  x_D4_M4 free
  x_D4_M5 free
Binaries
  y_D1_M1 y_D1_M2 y_D1_M3 y_D1_M4 y_D1_M5 y_D2_M1 y_D2_M2 y_D2_M3 y_D2_M4 y_D2_M5 y_D3_M1 y_D3_M2 y_D3_M3 y_D3_M4 y_D3_M5 y_D4_M1 y_D4_M2 y_D4_M3 y_D4_M4 y_D4_M5
End
