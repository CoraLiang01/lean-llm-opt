\ Canonical native-bound formulation; bounds excluded from WL graph.
Minimize
  obj: + 3000 y_W1 + 3200 y_W2 + 3100 y_W3 + 2800 y_W4 + 3500 y_W5 + 2700 y_W6 + 2900 y_W7 + 3050 y_W8 + 3100 y_W9 + 2200 y_W10 + 2890 y_W11 + 12 x_W1_S1 + 17 x_W1_S2 + 13 x_W1_S3 + 18 x_W1_S4 + 10 x_W1_S5 + 15 x_W1_S6 + 14 x_W1_S7 + 19 x_W1_S8 + 17 x_W1_S9 + 14 x_W1_S10 + 15 x_W1_S11 + 11 x_W2_S1 + 19 x_W2_S2 + 14 x_W2_S3 + 16 x_W2_S4 + 13 x_W2_S5 + 12 x_W2_S6 + 13 x_W2_S7 + 16 x_W2_S8 + 18 x_W2_S9 + 13 x_W2_S10 + 13 x_W2_S11 + 14 x_W3_S1 + 15 x_W3_S2 + 12 x_W3_S3 + 17 x_W3_S4 + 12 x_W3_S5 + 14 x_W3_S6 + 15 x_W3_S7 + 18 x_W3_S8 + 12 x_W3_S9 + 15 x_W3_S10 + 16 x_W3_S11 + 15 x_W4_S1 + 20 x_W4_S2 + 14 x_W4_S3 + 13 x_W4_S4 + 19 x_W4_S5 + 16 x_W4_S6 + 17 x_W4_S7 + 20 x_W4_S8 + 14 x_W4_S9 + 17 x_W4_S10 + 17 x_W4_S11 + 17 x_W5_S1 + 18 x_W5_S2 + 16 x_W5_S3 + 18 x_W5_S4 + 15 x_W5_S5 + 13 x_W5_S6 + 12 x_W5_S7 + 17 x_W5_S8 + 16 x_W5_S9
    + 16 x_W5_S10 + 11 x_W5_S11 + 13 x_W6_S1 + 14 x_W6_S2 + 15 x_W6_S3 + 17 x_W6_S4 + 11 x_W6_S5 + 17 x_W6_S6 + 13 x_W6_S7 + 19 x_W6_S8 + 15 x_W6_S9 + 18 x_W6_S10 + 13 x_W6_S11 + 12 x_W7_S1 + 17 x_W7_S2 + 11 x_W7_S3 + 14 x_W7_S4 + 12 x_W7_S5 + 16 x_W7_S6 + 14 x_W7_S7 + 16 x_W7_S8 + 14 x_W7_S9 + 14 x_W7_S10 + 14 x_W7_S11 + 16 x_W8_S1 + 15 x_W8_S2 + 14 x_W8_S3 + 19 x_W8_S4 + 14 x_W8_S5 + 16 x_W8_S6 + 15 x_W8_S7 + 18 x_W8_S8 + 17 x_W8_S9 + 19 x_W8_S10 + 15 x_W8_S11 + 16 x_W9_S1 + 13 x_W9_S2 + 16 x_W9_S3 + 16 x_W9_S4 + 12 x_W9_S5 + 14 x_W9_S6 + 12 x_W9_S7 + 15 x_W9_S8 + 21 x_W9_S9 + 15 x_W9_S10 + 19 x_W9_S11 + 14 x_W10_S1 + 15 x_W10_S2 + 18 x_W10_S3 + 13 x_W10_S4 + 15 x_W10_S5 + 18 x_W10_S6 + 16 x_W10_S7 + 15 x_W10_S8 + 15 x_W10_S9 + 17 x_W10_S10 + 21 x_W10_S11 + 15 x_W11_S1 + 16 x_W11_S2 + 17 x_W11_S3 + 18 x_W11_S4 + 17 x_W11_S5
    + 19 x_W11_S6 + 14 x_W11_S7 + 18 x_W11_S8 + 18 x_W11_S9 + 19 x_W11_S10 + 13 x_W11_S11
Subject To
  Demand_S1: + 1 x_W1_S1 + 1 x_W2_S1 + 1 x_W3_S1 + 1 x_W4_S1 + 1 x_W5_S1 + 1 x_W6_S1 + 1 x_W7_S1 + 1 x_W8_S1 + 1 x_W9_S1 + 1 x_W10_S1 + 1 x_W11_S1 = 30
  Demand_S2: + 1 x_W1_S2 + 1 x_W2_S2 + 1 x_W3_S2 + 1 x_W4_S2 + 1 x_W5_S2 + 1 x_W6_S2 + 1 x_W7_S2 + 1 x_W8_S2 + 1 x_W9_S2 + 1 x_W10_S2 + 1 x_W11_S2 = 40
  Demand_S3: + 1 x_W1_S3 + 1 x_W2_S3 + 1 x_W3_S3 + 1 x_W4_S3 + 1 x_W5_S3 + 1 x_W6_S3 + 1 x_W7_S3 + 1 x_W8_S3 + 1 x_W9_S3 + 1 x_W10_S3 + 1 x_W11_S3 = 20
  Demand_S4: + 1 x_W1_S4 + 1 x_W2_S4 + 1 x_W3_S4 + 1 x_W4_S4 + 1 x_W5_S4 + 1 x_W6_S4 + 1 x_W7_S4 + 1 x_W8_S4 + 1 x_W9_S4 + 1 x_W10_S4 + 1 x_W11_S4 = 35
  Demand_S5: + 1 x_W1_S5 + 1 x_W2_S5 + 1 x_W3_S5 + 1 x_W4_S5 + 1 x_W5_S5 + 1 x_W6_S5 + 1 x_W7_S5 + 1 x_W8_S5 + 1 x_W9_S5 + 1 x_W10_S5 + 1 x_W11_S5 = 20
  Demand_S6: + 1 x_W1_S6 + 1 x_W2_S6 + 1 x_W3_S6 + 1 x_W4_S6 + 1 x_W5_S6 + 1 x_W6_S6 + 1 x_W7_S6 + 1 x_W8_S6 + 1 x_W9_S6 + 1 x_W10_S6 + 1 x_W11_S6 = 25
  Demand_S7: + 1 x_W1_S7 + 1 x_W2_S7 + 1 x_W3_S7 + 1 x_W4_S7 + 1 x_W5_S7 + 1 x_W6_S7 + 1 x_W7_S7 + 1 x_W8_S7 + 1 x_W9_S7 + 1 x_W10_S7 + 1 x_W11_S7 = 45
  Demand_S8: + 1 x_W1_S8 + 1 x_W2_S8 + 1 x_W3_S8 + 1 x_W4_S8 + 1 x_W5_S8 + 1 x_W6_S8 + 1 x_W7_S8 + 1 x_W8_S8 + 1 x_W9_S8 + 1 x_W10_S8 + 1 x_W11_S8 = 38
  Demand_S9: + 1 x_W1_S9 + 1 x_W2_S9 + 1 x_W3_S9 + 1 x_W4_S9 + 1 x_W5_S9 + 1 x_W6_S9 + 1 x_W7_S9 + 1 x_W8_S9 + 1 x_W9_S9 + 1 x_W10_S9 + 1 x_W11_S9 = 32
  Demand_S10: + 1 x_W1_S10 + 1 x_W2_S10 + 1 x_W3_S10 + 1 x_W4_S10 + 1 x_W5_S10 + 1 x_W6_S10 + 1 x_W7_S10 + 1 x_W8_S10 + 1 x_W9_S10 + 1 x_W10_S10 + 1 x_W11_S10 = 41
  Demand_S11: + 1 x_W1_S11 + 1 x_W2_S11 + 1 x_W3_S11 + 1 x_W4_S11 + 1 x_W5_S11 + 1 x_W6_S11 + 1 x_W7_S11 + 1 x_W8_S11 + 1 x_W9_S11 + 1 x_W10_S11 + 1 x_W11_S11 = 44
  Capacity_W1: - 180 y_W1 + 1 x_W1_S1 + 1 x_W1_S2 + 1 x_W1_S3 + 1 x_W1_S4 + 1 x_W1_S5 + 1 x_W1_S6 + 1 x_W1_S7 + 1 x_W1_S8 + 1 x_W1_S9 + 1 x_W1_S10 + 1 x_W1_S11 <= 0
  Capacity_W2: - 160 y_W2 + 1 x_W2_S1 + 1 x_W2_S2 + 1 x_W2_S3 + 1 x_W2_S4 + 1 x_W2_S5 + 1 x_W2_S6 + 1 x_W2_S7 + 1 x_W2_S8 + 1 x_W2_S9 + 1 x_W2_S10 + 1 x_W2_S11 <= 0
  Capacity_W3: - 200 y_W3 + 1 x_W3_S1 + 1 x_W3_S2 + 1 x_W3_S3 + 1 x_W3_S4 + 1 x_W3_S5 + 1 x_W3_S6 + 1 x_W3_S7 + 1 x_W3_S8 + 1 x_W3_S9 + 1 x_W3_S10 + 1 x_W3_S11 <= 0
  Capacity_W4: - 150 y_W4 + 1 x_W4_S1 + 1 x_W4_S2 + 1 x_W4_S3 + 1 x_W4_S4 + 1 x_W4_S5 + 1 x_W4_S6 + 1 x_W4_S7 + 1 x_W4_S8 + 1 x_W4_S9 + 1 x_W4_S10 + 1 x_W4_S11 <= 0
  Capacity_W5: - 170 y_W5 + 1 x_W5_S1 + 1 x_W5_S2 + 1 x_W5_S3 + 1 x_W5_S4 + 1 x_W5_S5 + 1 x_W5_S6 + 1 x_W5_S7 + 1 x_W5_S8 + 1 x_W5_S9 + 1 x_W5_S10 + 1 x_W5_S11 <= 0
  Capacity_W6: - 190 y_W6 + 1 x_W6_S1 + 1 x_W6_S2 + 1 x_W6_S3 + 1 x_W6_S4 + 1 x_W6_S5 + 1 x_W6_S6 + 1 x_W6_S7 + 1 x_W6_S8 + 1 x_W6_S9 + 1 x_W6_S10 + 1 x_W6_S11 <= 0
  Capacity_W7: - 160 y_W7 + 1 x_W7_S1 + 1 x_W7_S2 + 1 x_W7_S3 + 1 x_W7_S4 + 1 x_W7_S5 + 1 x_W7_S6 + 1 x_W7_S7 + 1 x_W7_S8 + 1 x_W7_S9 + 1 x_W7_S10 + 1 x_W7_S11 <= 0
  Capacity_W8: - 175 y_W8 + 1 x_W8_S1 + 1 x_W8_S2 + 1 x_W8_S3 + 1 x_W8_S4 + 1 x_W8_S5 + 1 x_W8_S6 + 1 x_W8_S7 + 1 x_W8_S8 + 1 x_W8_S9 + 1 x_W8_S10 + 1 x_W8_S11 <= 0
  Capacity_W9: - 170 y_W9 + 1 x_W9_S1 + 1 x_W9_S2 + 1 x_W9_S3 + 1 x_W9_S4 + 1 x_W9_S5 + 1 x_W9_S6 + 1 x_W9_S7 + 1 x_W9_S8 + 1 x_W9_S9 + 1 x_W9_S10 + 1 x_W9_S11 <= 0
  Capacity_W10: - 180 y_W10 + 1 x_W10_S1 + 1 x_W10_S2 + 1 x_W10_S3 + 1 x_W10_S4 + 1 x_W10_S5 + 1 x_W10_S6 + 1 x_W10_S7 + 1 x_W10_S8 + 1 x_W10_S9 + 1 x_W10_S10 + 1 x_W10_S11 <= 0
  Capacity_W11: - 190 y_W11 + 1 x_W11_S1 + 1 x_W11_S2 + 1 x_W11_S3 + 1 x_W11_S4 + 1 x_W11_S5 + 1 x_W11_S6 + 1 x_W11_S7 + 1 x_W11_S8 + 1 x_W11_S9 + 1 x_W11_S10 + 1 x_W11_S11 <= 0
Bounds
  0 <= y_W1 <= 1
  0 <= y_W2 <= 1
  0 <= y_W3 <= 1
  0 <= y_W4 <= 1
  0 <= y_W5 <= 1
  0 <= y_W6 <= 1
  0 <= y_W7 <= 1
  0 <= y_W8 <= 1
  0 <= y_W9 <= 1
  0 <= y_W10 <= 1
  0 <= y_W11 <= 1
  0 <= x_W1_S1 <= +inf
  0 <= x_W1_S2 <= +inf
  0 <= x_W1_S3 <= +inf
  0 <= x_W1_S4 <= +inf
  0 <= x_W1_S5 <= +inf
  0 <= x_W1_S6 <= +inf
  0 <= x_W1_S7 <= +inf
  0 <= x_W1_S8 <= +inf
  0 <= x_W1_S9 <= +inf
  0 <= x_W1_S10 <= +inf
  0 <= x_W1_S11 <= +inf
  0 <= x_W2_S1 <= +inf
  0 <= x_W2_S2 <= +inf
  0 <= x_W2_S3 <= +inf
  0 <= x_W2_S4 <= +inf
  0 <= x_W2_S5 <= +inf
  0 <= x_W2_S6 <= +inf
  0 <= x_W2_S7 <= +inf
  0 <= x_W2_S8 <= +inf
  0 <= x_W2_S9 <= +inf
  0 <= x_W2_S10 <= +inf
  0 <= x_W2_S11 <= +inf
  0 <= x_W3_S1 <= +inf
  0 <= x_W3_S2 <= +inf
  0 <= x_W3_S3 <= +inf
  0 <= x_W3_S4 <= +inf
  0 <= x_W3_S5 <= +inf
  0 <= x_W3_S6 <= +inf
  0 <= x_W3_S7 <= +inf
  0 <= x_W3_S8 <= +inf
  0 <= x_W3_S9 <= +inf
  0 <= x_W3_S10 <= +inf
  0 <= x_W3_S11 <= +inf
  0 <= x_W4_S1 <= +inf
  0 <= x_W4_S2 <= +inf
  0 <= x_W4_S3 <= +inf
  0 <= x_W4_S4 <= +inf
  0 <= x_W4_S5 <= +inf
  0 <= x_W4_S6 <= +inf
  0 <= x_W4_S7 <= +inf
  0 <= x_W4_S8 <= +inf
  0 <= x_W4_S9 <= +inf
  0 <= x_W4_S10 <= +inf
  0 <= x_W4_S11 <= +inf
  0 <= x_W5_S1 <= +inf
  0 <= x_W5_S2 <= +inf
  0 <= x_W5_S3 <= +inf
  0 <= x_W5_S4 <= +inf
  0 <= x_W5_S5 <= +inf
  0 <= x_W5_S6 <= +inf
  0 <= x_W5_S7 <= +inf
  0 <= x_W5_S8 <= +inf
  0 <= x_W5_S9 <= +inf
  0 <= x_W5_S10 <= +inf
  0 <= x_W5_S11 <= +inf
  0 <= x_W6_S1 <= +inf
  0 <= x_W6_S2 <= +inf
  0 <= x_W6_S3 <= +inf
  0 <= x_W6_S4 <= +inf
  0 <= x_W6_S5 <= +inf
  0 <= x_W6_S6 <= +inf
  0 <= x_W6_S7 <= +inf
  0 <= x_W6_S8 <= +inf
  0 <= x_W6_S9 <= +inf
  0 <= x_W6_S10 <= +inf
  0 <= x_W6_S11 <= +inf
  0 <= x_W7_S1 <= +inf
  0 <= x_W7_S2 <= +inf
  0 <= x_W7_S3 <= +inf
  0 <= x_W7_S4 <= +inf
  0 <= x_W7_S5 <= +inf
  0 <= x_W7_S6 <= +inf
  0 <= x_W7_S7 <= +inf
  0 <= x_W7_S8 <= +inf
  0 <= x_W7_S9 <= +inf
  0 <= x_W7_S10 <= +inf
  0 <= x_W7_S11 <= +inf
  0 <= x_W8_S1 <= +inf
  0 <= x_W8_S2 <= +inf
  0 <= x_W8_S3 <= +inf
  0 <= x_W8_S4 <= +inf
  0 <= x_W8_S5 <= +inf
  0 <= x_W8_S6 <= +inf
  0 <= x_W8_S7 <= +inf
  0 <= x_W8_S8 <= +inf
  0 <= x_W8_S9 <= +inf
  0 <= x_W8_S10 <= +inf
  0 <= x_W8_S11 <= +inf
  0 <= x_W9_S1 <= +inf
  0 <= x_W9_S2 <= +inf
  0 <= x_W9_S3 <= +inf
  0 <= x_W9_S4 <= +inf
  0 <= x_W9_S5 <= +inf
  0 <= x_W9_S6 <= +inf
  0 <= x_W9_S7 <= +inf
  0 <= x_W9_S8 <= +inf
  0 <= x_W9_S9 <= +inf
  0 <= x_W9_S10 <= +inf
  0 <= x_W9_S11 <= +inf
  0 <= x_W10_S1 <= +inf
  0 <= x_W10_S2 <= +inf
  0 <= x_W10_S3 <= +inf
  0 <= x_W10_S4 <= +inf
  0 <= x_W10_S5 <= +inf
  0 <= x_W10_S6 <= +inf
  0 <= x_W10_S7 <= +inf
  0 <= x_W10_S8 <= +inf
  0 <= x_W10_S9 <= +inf
  0 <= x_W10_S10 <= +inf
  0 <= x_W10_S11 <= +inf
  0 <= x_W11_S1 <= +inf
  0 <= x_W11_S2 <= +inf
  0 <= x_W11_S3 <= +inf
  0 <= x_W11_S4 <= +inf
  0 <= x_W11_S5 <= +inf
  0 <= x_W11_S6 <= +inf
  0 <= x_W11_S7 <= +inf
  0 <= x_W11_S8 <= +inf
  0 <= x_W11_S9 <= +inf
  0 <= x_W11_S10 <= +inf
  0 <= x_W11_S11 <= +inf
Binaries
  y_W1 y_W2 y_W3 y_W4 y_W5 y_W6 y_W7 y_W8 y_W9 y_W10 y_W11
End
