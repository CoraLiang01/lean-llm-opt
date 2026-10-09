\ Canonical native-bound formulation; bounds excluded from WL graph.
Minimize
  obj: + 17 x_M01 + 760 y_M01 + 1 inv_M01 + 18 x_M02 + 820 y_M02 + 1.1000000000000001 inv_M02 + 20 x_M03 + 930 y_M03 + 1.2 inv_M03 + 19 x_M04 + 780 y_M04 + 1 inv_M04 + 21 x_M05 + 960 y_M05 + 1.3 inv_M05 + 22 x_M06 + 1040 y_M06 + 1.3999999999999999 inv_M06 + 18 x_M07 + 800 y_M07 + 1.1000000000000001 inv_M07 + 20 x_M08 + 900 y_M08 + 1.2 inv_M08 + 17 x_M09 + 740 y_M09 + 1 inv_M09 + 23 x_M10 + 1100 y_M10 + 1.5 inv_M10 + 19 x_M11 + 850 y_M11 + 1.1000000000000001 inv_M11 + 21 x_M12 + 980 y_M12
Subject To
  demand_balance_M01: + 1 x_M01 - 1 inv_M01 = 95
  demand_balance_M02: + 1 inv_M01 + 1 x_M02 - 1 inv_M02 = 70
  demand_balance_M03: + 1 inv_M02 + 1 x_M03 - 1 inv_M03 = 130
  demand_balance_M04: + 1 inv_M03 + 1 x_M04 - 1 inv_M04 = 85
  demand_balance_M05: + 1 inv_M04 + 1 x_M05 - 1 inv_M05 = 115
  demand_balance_M06: + 1 inv_M05 + 1 x_M06 - 1 inv_M06 = 140
  demand_balance_M07: + 1 inv_M06 + 1 x_M07 - 1 inv_M07 = 90
  demand_balance_M08: + 1 inv_M07 + 1 x_M08 - 1 inv_M08 = 125
  demand_balance_M09: + 1 inv_M08 + 1 x_M09 - 1 inv_M09 = 75
  demand_balance_M10: + 1 inv_M09 + 1 x_M10 - 1 inv_M10 = 150
  demand_balance_M11: + 1 inv_M10 + 1 x_M11 - 1 inv_M11 = 105
  demand_balance_M12: + 1 inv_M11 + 1 x_M12 - 1 inv_M12 = 120
  setup_capacity_M01: + 1 x_M01 - 310 y_M01 <= 0
  setup_capacity_M02: + 1 x_M02 - 260 y_M02 <= 0
  setup_capacity_M03: + 1 x_M03 - 320 y_M03 <= 0
  setup_capacity_M04: + 1 x_M04 - 300 y_M04 <= 0
  setup_capacity_M05: + 1 x_M05 - 340 y_M05 <= 0
  setup_capacity_M06: + 1 x_M06 - 360 y_M06 <= 0
  setup_capacity_M07: + 1 x_M07 - 290 y_M07 <= 0
  setup_capacity_M08: + 1 x_M08 - 330 y_M08 <= 0
  setup_capacity_M09: + 1 x_M09 - 280 y_M09 <= 0
  setup_capacity_M10: + 1 x_M10 - 370 y_M10 <= 0
  setup_capacity_M11: + 1 x_M11 - 310 y_M11 <= 0
  setup_capacity_M12: + 1 x_M12 - 350 y_M12 <= 0
Bounds
  0 <= x_M01 <= +inf
  0 <= y_M01 <= 1
  0 <= inv_M01 <= +inf
  0 <= x_M02 <= +inf
  0 <= y_M02 <= 1
  0 <= inv_M02 <= +inf
  0 <= x_M03 <= +inf
  0 <= y_M03 <= 1
  0 <= inv_M03 <= +inf
  0 <= x_M04 <= +inf
  0 <= y_M04 <= 1
  0 <= inv_M04 <= +inf
  0 <= x_M05 <= +inf
  0 <= y_M05 <= 1
  0 <= inv_M05 <= +inf
  0 <= x_M06 <= +inf
  0 <= y_M06 <= 1
  0 <= inv_M06 <= +inf
  0 <= x_M07 <= +inf
  0 <= y_M07 <= 1
  0 <= inv_M07 <= +inf
  0 <= x_M08 <= +inf
  0 <= y_M08 <= 1
  0 <= inv_M08 <= +inf
  0 <= x_M09 <= +inf
  0 <= y_M09 <= 1
  0 <= inv_M09 <= +inf
  0 <= x_M10 <= +inf
  0 <= y_M10 <= 1
  0 <= inv_M10 <= +inf
  0 <= x_M11 <= +inf
  0 <= y_M11 <= 1
  0 <= inv_M11 <= +inf
  0 <= x_M12 <= +inf
  0 <= y_M12 <= 1
  inv_M12 = 0
Binaries
  y_M01 y_M02 y_M03 y_M04 y_M05 y_M06 y_M07 y_M08 y_M09 y_M10 y_M11 y_M12
End
