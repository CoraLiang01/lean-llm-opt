\ Canonical native-bound formulation; bounds excluded from WL graph.
Minimize
  obj: + 1 x[0] + 1 x[1] + 1 x[2] + 1 x[3] + 1 x[4] + 1 x[5] + 1 x[6] + 1 x[7] + 1 x[8] + 1 x[9] + 1 x[10] + 1 x[11] + 1 x[12] + 1 x[13] + 1 x[14] + 1 x[15] + 1 x[16] + 1 x[17] + 1 x[18] + 1 x[19] + 1 x[20] + 1 x[21] + 1 x[22] + 1 x[23]
Subject To
  Demand_Hour_00_Met: + 1 x[0] + 1 x[21] + 1 x[22] + 1 x[23] >= 20
  Demand_Hour_01_Met: + 1 x[0] + 1 x[1] + 1 x[22] + 1 x[23] >= 18
  Demand_Hour_02_Met: + 1 x[0] + 1 x[1] + 1 x[2] + 1 x[23] >= 15
  Demand_Hour_03_Met: + 1 x[0] + 1 x[1] + 1 x[2] + 1 x[3] >= 15
  Demand_Hour_04_Met: + 1 x[1] + 1 x[2] + 1 x[3] + 1 x[4] >= 20
  Demand_Hour_05_Met: + 1 x[2] + 1 x[3] + 1 x[4] + 1 x[5] >= 30
  Demand_Hour_06_Met: + 1 x[3] + 1 x[4] + 1 x[5] + 1 x[6] >= 60
  Demand_Hour_07_Met: + 1 x[4] + 1 x[5] + 1 x[6] + 1 x[7] >= 70
  Demand_Hour_08_Met: + 1 x[5] + 1 x[6] + 1 x[7] + 1 x[8] >= 50
  Demand_Hour_09_Met: + 1 x[6] + 1 x[7] + 1 x[8] + 1 x[9] >= 55
  Demand_Hour_10_Met: + 1 x[7] + 1 x[8] + 1 x[9] + 1 x[10] >= 65
  Demand_Hour_11_Met: + 1 x[8] + 1 x[9] + 1 x[10] + 1 x[11] >= 75
  Demand_Hour_12_Met: + 1 x[9] + 1 x[10] + 1 x[11] + 1 x[12] >= 80
  Demand_Hour_13_Met: + 1 x[10] + 1 x[11] + 1 x[12] + 1 x[13] >= 70
  Demand_Hour_14_Met: + 1 x[11] + 1 x[12] + 1 x[13] + 1 x[14] >= 60
  Demand_Hour_15_Met: + 1 x[12] + 1 x[13] + 1 x[14] + 1 x[15] >= 55
  Demand_Hour_16_Met: + 1 x[13] + 1 x[14] + 1 x[15] + 1 x[16] >= 60
  Demand_Hour_17_Met: + 1 x[14] + 1 x[15] + 1 x[16] + 1 x[17] >= 75
  Demand_Hour_18_Met: + 1 x[15] + 1 x[16] + 1 x[17] + 1 x[18] >= 85
  Demand_Hour_19_Met: + 1 x[16] + 1 x[17] + 1 x[18] + 1 x[19] >= 70
  Demand_Hour_20_Met: + 1 x[17] + 1 x[18] + 1 x[19] + 1 x[20] >= 50
  Demand_Hour_21_Met: + 1 x[18] + 1 x[19] + 1 x[20] + 1 x[21] >= 40
  Demand_Hour_22_Met: + 1 x[19] + 1 x[20] + 1 x[21] + 1 x[22] >= 35
  Demand_Hour_23_Met: + 1 x[20] + 1 x[21] + 1 x[22] + 1 x[23] >= 25
Bounds
  0 <= x[0] <= +inf
  0 <= x[1] <= +inf
  0 <= x[2] <= +inf
  0 <= x[3] <= +inf
  0 <= x[4] <= +inf
  0 <= x[5] <= +inf
  0 <= x[6] <= +inf
  0 <= x[7] <= +inf
  0 <= x[8] <= +inf
  0 <= x[9] <= +inf
  0 <= x[10] <= +inf
  0 <= x[11] <= +inf
  0 <= x[12] <= +inf
  0 <= x[13] <= +inf
  0 <= x[14] <= +inf
  0 <= x[15] <= +inf
  0 <= x[16] <= +inf
  0 <= x[17] <= +inf
  0 <= x[18] <= +inf
  0 <= x[19] <= +inf
  0 <= x[20] <= +inf
  0 <= x[21] <= +inf
  0 <= x[22] <= +inf
  0 <= x[23] <= +inf
Generals
  x[0] x[1] x[2] x[3] x[4] x[5] x[6] x[7] x[8] x[9] x[10] x[11] x[12] x[13] x[14] x[15] x[16] x[17] x[18] x[19] x[20] x[21] x[22] x[23]
End
