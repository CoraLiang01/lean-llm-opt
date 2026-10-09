\ Canonical native-bound formulation; bounds excluded from WL graph.
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
Bounds
  0 <= x_D1_M1 <= +inf
  0 <= y_D1_M1 <= 1
  0 <= x_D1_M2 <= +inf
  0 <= y_D1_M2 <= 1
  0 <= x_D1_M3 <= +inf
  0 <= y_D1_M3 <= 1
  0 <= x_D1_M4 <= +inf
  0 <= y_D1_M4 <= 1
  0 <= x_D1_M5 <= +inf
  0 <= y_D1_M5 <= 1
  0 <= x_D2_M1 <= +inf
  0 <= y_D2_M1 <= 1
  0 <= x_D2_M2 <= +inf
  0 <= y_D2_M2 <= 1
  0 <= x_D2_M3 <= +inf
  0 <= y_D2_M3 <= 1
  0 <= x_D2_M4 <= +inf
  0 <= y_D2_M4 <= 1
  0 <= x_D2_M5 <= +inf
  0 <= y_D2_M5 <= 1
  0 <= x_D3_M1 <= +inf
  0 <= y_D3_M1 <= 1
  0 <= x_D3_M2 <= +inf
  0 <= y_D3_M2 <= 1
  0 <= x_D3_M3 <= +inf
  0 <= y_D3_M3 <= 1
  0 <= x_D3_M4 <= +inf
  0 <= y_D3_M4 <= 1
  0 <= x_D3_M5 <= +inf
  0 <= y_D3_M5 <= 1
  0 <= x_D4_M1 <= +inf
  0 <= y_D4_M1 <= 1
  0 <= x_D4_M2 <= +inf
  0 <= y_D4_M2 <= 1
  0 <= x_D4_M3 <= +inf
  0 <= y_D4_M3 <= 1
  0 <= x_D4_M4 <= +inf
  0 <= y_D4_M4 <= 1
  0 <= x_D4_M5 <= +inf
  0 <= y_D4_M5 <= 1
Binaries
  y_D1_M1 y_D1_M2 y_D1_M3 y_D1_M4 y_D1_M5 y_D2_M1 y_D2_M2 y_D2_M3 y_D2_M4 y_D2_M5 y_D3_M1 y_D3_M2 y_D3_M3 y_D3_M4 y_D3_M5 y_D4_M1 y_D4_M2 y_D4_M3 y_D4_M4 y_D4_M5
End
