\ Canonical native-bound formulation; bounds excluded from WL graph.
Minimize
  obj: + 18 x_M01 + 900 y_M01 + 1.1000000000000001 inv_M01 + 20 x_M02 + 1100 y_M02 + 1.2 inv_M02 + 19 x_M03 + 950 y_M03 + 1 inv_M03 + 21 x_M04 + 1200 y_M04 + 1.3 inv_M04 + 18 x_M05 + 850 y_M05 + 1.1000000000000001 inv_M05 + 22 x_M06 + 1300 y_M06 + 1.3999999999999999 inv_M06 + 20 x_M07 + 1000 y_M07 + 1.2 inv_M07 + 19 x_M08 + 900 y_M08 + 1 inv_M08 + 23 x_M09 + 1400 y_M09 + 1.5 inv_M09 + 21 x_M10 + 1150 y_M10 + 1.2 inv_M10 + 20 x_M11 + 1000 y_M11 + 1.1000000000000001 inv_M11 + 18 x_M12 + 850 y_M12 + 1 inv_M12 + 19 x_M13 + 950 y_M13 + 1.1000000000000001 inv_M13 + 22 x_M14 + 1250 y_M14 + 1.3999999999999999 inv_M14 + 21 x_M15 + 1150 y_M15 + 1.2 inv_M15 + 20 x_M16 + 1050 y_M16 + 1.1000000000000001 inv_M16 + 18 x_M17 + 800 y_M17 + 1 inv_M17 + 23 x_M18 + 1350 y_M18 + 1.5 inv_M18 + 21 x_M19 + 1100 y_M19 + 1.2 inv_M19 + 19 x_M20 + 950 y_M20
    + 1.1000000000000001 inv_M20 + 22 x_M21 + 1200 y_M21 + 1.3 inv_M21 + 18 x_M22 + 850 y_M22 + 1 inv_M22 + 20 x_M23 + 1000 y_M23 + 1.2 inv_M23 + 21 x_M24 + 1100 y_M24
Subject To
  demand_balance_M01: + 1 x_M01 - 1 inv_M01 = 80
  demand_balance_M02: + 1 inv_M01 + 1 x_M02 - 1 inv_M02 = 120
  demand_balance_M03: + 1 inv_M02 + 1 x_M03 - 1 inv_M03 = 60
  demand_balance_M04: + 1 inv_M03 + 1 x_M04 - 1 inv_M04 = 150
  demand_balance_M05: + 1 inv_M04 + 1 x_M05 - 1 inv_M05 = 90
  demand_balance_M06: + 1 inv_M05 + 1 x_M06 - 1 inv_M06 = 110
  demand_balance_M07: + 1 inv_M06 + 1 x_M07 - 1 inv_M07 = 140
  demand_balance_M08: + 1 inv_M07 + 1 x_M08 - 1 inv_M08 = 70
  demand_balance_M09: + 1 inv_M08 + 1 x_M09 - 1 inv_M09 = 160
  demand_balance_M10: + 1 inv_M09 + 1 x_M10 - 1 inv_M10 = 100
  demand_balance_M11: + 1 inv_M10 + 1 x_M11 - 1 inv_M11 = 130
  demand_balance_M12: + 1 inv_M11 + 1 x_M12 - 1 inv_M12 = 85
  demand_balance_M13: + 1 inv_M12 + 1 x_M13 - 1 inv_M13 = 115
  demand_balance_M14: + 1 inv_M13 + 1 x_M14 - 1 inv_M14 = 95
  demand_balance_M15: + 1 inv_M14 + 1 x_M15 - 1 inv_M15 = 145
  demand_balance_M16: + 1 inv_M15 + 1 x_M16 - 1 inv_M16 = 105
  demand_balance_M17: + 1 inv_M16 + 1 x_M17 - 1 inv_M17 = 75
  demand_balance_M18: + 1 inv_M17 + 1 x_M18 - 1 inv_M18 = 155
  demand_balance_M19: + 1 inv_M18 + 1 x_M19 - 1 inv_M19 = 125
  demand_balance_M20: + 1 inv_M19 + 1 x_M20 - 1 inv_M20 = 90
  demand_balance_M21: + 1 inv_M20 + 1 x_M21 - 1 inv_M21 = 135
  demand_balance_M22: + 1 inv_M21 + 1 x_M22 - 1 inv_M22 = 65
  demand_balance_M23: + 1 inv_M22 + 1 x_M23 - 1 inv_M23 = 150
  demand_balance_M24: + 1 inv_M23 + 1 x_M24 - 1 inv_M24 = 110
  setup_capacity_M01: + 1 x_M01 - 420 y_M01 <= 0
  setup_capacity_M02: + 1 x_M02 - 360 y_M02 <= 0
  setup_capacity_M03: + 1 x_M03 - 380 y_M03 <= 0
  setup_capacity_M04: + 1 x_M04 - 420 y_M04 <= 0
  setup_capacity_M05: + 1 x_M05 - 350 y_M05 <= 0
  setup_capacity_M06: + 1 x_M06 - 430 y_M06 <= 0
  setup_capacity_M07: + 1 x_M07 - 390 y_M07 <= 0
  setup_capacity_M08: + 1 x_M08 - 360 y_M08 <= 0
  setup_capacity_M09: + 1 x_M09 - 440 y_M09 <= 0
  setup_capacity_M10: + 1 x_M10 - 400 y_M10 <= 0
  setup_capacity_M11: + 1 x_M11 - 380 y_M11 <= 0
  setup_capacity_M12: + 1 x_M12 - 350 y_M12 <= 0
  setup_capacity_M13: + 1 x_M13 - 390 y_M13 <= 0
  setup_capacity_M14: + 1 x_M14 - 430 y_M14 <= 0
  setup_capacity_M15: + 1 x_M15 - 410 y_M15 <= 0
  setup_capacity_M16: + 1 x_M16 - 370 y_M16 <= 0
  setup_capacity_M17: + 1 x_M17 - 360 y_M17 <= 0
  setup_capacity_M18: + 1 x_M18 - 440 y_M18 <= 0
  setup_capacity_M19: + 1 x_M19 - 400 y_M19 <= 0
  setup_capacity_M20: + 1 x_M20 - 360 y_M20 <= 0
  setup_capacity_M21: + 1 x_M21 - 420 y_M21 <= 0
  setup_capacity_M22: + 1 x_M22 - 350 y_M22 <= 0
  setup_capacity_M23: + 1 x_M23 - 430 y_M23 <= 0
  setup_capacity_M24: + 1 x_M24 - 390 y_M24 <= 0
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
  0 <= inv_M12 <= +inf
  0 <= x_M13 <= +inf
  0 <= y_M13 <= 1
  0 <= inv_M13 <= +inf
  0 <= x_M14 <= +inf
  0 <= y_M14 <= 1
  0 <= inv_M14 <= +inf
  0 <= x_M15 <= +inf
  0 <= y_M15 <= 1
  0 <= inv_M15 <= +inf
  0 <= x_M16 <= +inf
  0 <= y_M16 <= 1
  0 <= inv_M16 <= +inf
  0 <= x_M17 <= +inf
  0 <= y_M17 <= 1
  0 <= inv_M17 <= +inf
  0 <= x_M18 <= +inf
  0 <= y_M18 <= 1
  0 <= inv_M18 <= +inf
  0 <= x_M19 <= +inf
  0 <= y_M19 <= 1
  0 <= inv_M19 <= +inf
  0 <= x_M20 <= +inf
  0 <= y_M20 <= 1
  0 <= inv_M20 <= +inf
  0 <= x_M21 <= +inf
  0 <= y_M21 <= 1
  0 <= inv_M21 <= +inf
  0 <= x_M22 <= +inf
  0 <= y_M22 <= 1
  0 <= inv_M22 <= +inf
  0 <= x_M23 <= +inf
  0 <= y_M23 <= 1
  0 <= inv_M23 <= +inf
  0 <= x_M24 <= +inf
  0 <= y_M24 <= 1
  inv_M24 = 0
Binaries
  y_M01 y_M02 y_M03 y_M04 y_M05 y_M06 y_M07 y_M08 y_M09 y_M10 y_M11 y_M12 y_M13 y_M14 y_M15 y_M16 y_M17 y_M18 y_M19 y_M20 y_M21 y_M22 y_M23 y_M24
End
