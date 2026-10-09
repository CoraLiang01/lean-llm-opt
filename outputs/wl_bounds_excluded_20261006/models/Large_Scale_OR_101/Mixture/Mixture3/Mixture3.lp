\ Canonical native-bound formulation; bounds excluded from WL graph.
Maximize
  obj: + 0.75 x_A1_I + 1.1499999999999999 x_A1_II + 0.77529999999999999 x_A2_I + 1.3611 x_A2_II + 1.9148000000000001 x_A2_III - 0.375 x_B1_I - 0.5 x_B1_II - 0.44742857142857145 x_B2_I - 1.2304285714285714 x_B2_III - 0.34999999999999998 x_B3_I
Subject To
  capacity_A1: + 5 x_A1_I + 10 x_A1_II <= 6000
  capacity_A2: + 7 x_A2_I + 9 x_A2_II + 12 x_A2_III <= 10000
  capacity_B1: + 6 x_B1_I + 8 x_B1_II <= 4000
  capacity_B2: + 4 x_B2_I + 11 x_B2_III <= 7000
  flow_I: + 1 x_A1_I + 1 x_A2_I - 1 x_B1_I - 1 x_B2_I - 1 x_B3_I = 0
  flow_II: + 1 x_A1_II + 1 x_A2_II - 1 x_B1_II = 0
  flow_III: + 1 x_A2_III - 1 x_B2_III = 0
Bounds
  0 <= x_A1_I <= +inf
  0 <= x_A1_II <= +inf
  0 <= x_A2_I <= +inf
  0 <= x_A2_II <= +inf
  0 <= x_A2_III <= +inf
  0 <= x_B1_I <= +inf
  0 <= x_B1_II <= +inf
  0 <= x_B2_I <= +inf
  0 <= x_B2_III <= +inf
  0 <= x_B3_I <= 571.42857142857144
End
