\ Canonical native-bound formulation; bounds excluded from WL graph.
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
Bounds
  0 <= x_S1_C1 <= +inf
  0 <= y_S1_C1 <= 1
  0 <= x_S1_C2 <= +inf
  0 <= y_S1_C2 <= 1
  0 <= x_S1_C3 <= +inf
  0 <= y_S1_C3 <= 1
  0 <= x_S1_C4 <= +inf
  0 <= y_S1_C4 <= 1
  0 <= x_S1_C5 <= +inf
  0 <= y_S1_C5 <= 1
  0 <= x_S1_C6 <= +inf
  0 <= y_S1_C6 <= 1
  0 <= x_S1_C7 <= +inf
  0 <= y_S1_C7 <= 1
  0 <= x_S1_C8 <= +inf
  0 <= y_S1_C8 <= 1
  0 <= x_S1_C9 <= +inf
  0 <= y_S1_C9 <= 1
  0 <= x_S1_C10 <= +inf
  0 <= y_S1_C10 <= 1
  0 <= x_S1_C11 <= +inf
  0 <= y_S1_C11 <= 1
  0 <= x_S1_C12 <= +inf
  0 <= y_S1_C12 <= 1
  0 <= x_S2_C1 <= +inf
  0 <= y_S2_C1 <= 1
  0 <= x_S2_C2 <= +inf
  0 <= y_S2_C2 <= 1
  0 <= x_S2_C3 <= +inf
  0 <= y_S2_C3 <= 1
  0 <= x_S2_C4 <= +inf
  0 <= y_S2_C4 <= 1
  0 <= x_S2_C5 <= +inf
  0 <= y_S2_C5 <= 1
  0 <= x_S2_C6 <= +inf
  0 <= y_S2_C6 <= 1
  0 <= x_S2_C7 <= +inf
  0 <= y_S2_C7 <= 1
  0 <= x_S2_C8 <= +inf
  0 <= y_S2_C8 <= 1
  0 <= x_S2_C9 <= +inf
  0 <= y_S2_C9 <= 1
  0 <= x_S2_C10 <= +inf
  0 <= y_S2_C10 <= 1
  0 <= x_S2_C11 <= +inf
  0 <= y_S2_C11 <= 1
  0 <= x_S2_C12 <= +inf
  0 <= y_S2_C12 <= 1
  0 <= x_S3_C1 <= +inf
  0 <= y_S3_C1 <= 1
  0 <= x_S3_C2 <= +inf
  0 <= y_S3_C2 <= 1
  0 <= x_S3_C3 <= +inf
  0 <= y_S3_C3 <= 1
  0 <= x_S3_C4 <= +inf
  0 <= y_S3_C4 <= 1
  0 <= x_S3_C5 <= +inf
  0 <= y_S3_C5 <= 1
  0 <= x_S3_C6 <= +inf
  0 <= y_S3_C6 <= 1
  0 <= x_S3_C7 <= +inf
  0 <= y_S3_C7 <= 1
  0 <= x_S3_C8 <= +inf
  0 <= y_S3_C8 <= 1
  0 <= x_S3_C9 <= +inf
  0 <= y_S3_C9 <= 1
  0 <= x_S3_C10 <= +inf
  0 <= y_S3_C10 <= 1
  0 <= x_S3_C11 <= +inf
  0 <= y_S3_C11 <= 1
  0 <= x_S3_C12 <= +inf
  0 <= y_S3_C12 <= 1
  0 <= x_S4_C1 <= +inf
  0 <= y_S4_C1 <= 1
  0 <= x_S4_C2 <= +inf
  0 <= y_S4_C2 <= 1
  0 <= x_S4_C3 <= +inf
  0 <= y_S4_C3 <= 1
  0 <= x_S4_C4 <= +inf
  0 <= y_S4_C4 <= 1
  0 <= x_S4_C5 <= +inf
  0 <= y_S4_C5 <= 1
  0 <= x_S4_C6 <= +inf
  0 <= y_S4_C6 <= 1
  0 <= x_S4_C7 <= +inf
  0 <= y_S4_C7 <= 1
  0 <= x_S4_C8 <= +inf
  0 <= y_S4_C8 <= 1
  0 <= x_S4_C9 <= +inf
  0 <= y_S4_C9 <= 1
  0 <= x_S4_C10 <= +inf
  0 <= y_S4_C10 <= 1
  0 <= x_S4_C11 <= +inf
  0 <= y_S4_C11 <= 1
  0 <= x_S4_C12 <= +inf
  0 <= y_S4_C12 <= 1
  0 <= x_S5_C1 <= +inf
  0 <= y_S5_C1 <= 1
  0 <= x_S5_C2 <= +inf
  0 <= y_S5_C2 <= 1
  0 <= x_S5_C3 <= +inf
  0 <= y_S5_C3 <= 1
  0 <= x_S5_C4 <= +inf
  0 <= y_S5_C4 <= 1
  0 <= x_S5_C5 <= +inf
  0 <= y_S5_C5 <= 1
  0 <= x_S5_C6 <= +inf
  0 <= y_S5_C6 <= 1
  0 <= x_S5_C7 <= +inf
  0 <= y_S5_C7 <= 1
  0 <= x_S5_C8 <= +inf
  0 <= y_S5_C8 <= 1
  0 <= x_S5_C9 <= +inf
  0 <= y_S5_C9 <= 1
  0 <= x_S5_C10 <= +inf
  0 <= y_S5_C10 <= 1
  0 <= x_S5_C11 <= +inf
  0 <= y_S5_C11 <= 1
  0 <= x_S5_C12 <= +inf
  0 <= y_S5_C12 <= 1
  0 <= x_S6_C1 <= +inf
  0 <= y_S6_C1 <= 1
  0 <= x_S6_C2 <= +inf
  0 <= y_S6_C2 <= 1
  0 <= x_S6_C3 <= +inf
  0 <= y_S6_C3 <= 1
  0 <= x_S6_C4 <= +inf
  0 <= y_S6_C4 <= 1
  0 <= x_S6_C5 <= +inf
  0 <= y_S6_C5 <= 1
  0 <= x_S6_C6 <= +inf
  0 <= y_S6_C6 <= 1
  0 <= x_S6_C7 <= +inf
  0 <= y_S6_C7 <= 1
  0 <= x_S6_C8 <= +inf
  0 <= y_S6_C8 <= 1
  0 <= x_S6_C9 <= +inf
  0 <= y_S6_C9 <= 1
  0 <= x_S6_C10 <= +inf
  0 <= y_S6_C10 <= 1
  0 <= x_S6_C11 <= +inf
  0 <= y_S6_C11 <= 1
  0 <= x_S6_C12 <= +inf
  0 <= y_S6_C12 <= 1
Binaries
  y_S1_C1 y_S1_C2 y_S1_C3 y_S1_C4 y_S1_C5 y_S1_C6 y_S1_C7 y_S1_C8 y_S1_C9 y_S1_C10 y_S1_C11 y_S1_C12 y_S2_C1 y_S2_C2 y_S2_C3 y_S2_C4 y_S2_C5 y_S2_C6 y_S2_C7 y_S2_C8 y_S2_C9 y_S2_C10 y_S2_C11 y_S2_C12 y_S3_C1 y_S3_C2 y_S3_C3 y_S3_C4 y_S3_C5 y_S3_C6 y_S3_C7 y_S3_C8 y_S3_C9 y_S3_C10 y_S3_C11 y_S3_C12 y_S4_C1 y_S4_C2 y_S4_C3 y_S4_C4 y_S4_C5 y_S4_C6 y_S4_C7 y_S4_C8 y_S4_C9 y_S4_C10 y_S4_C11 y_S4_C12 y_S5_C1 y_S5_C2 y_S5_C3 y_S5_C4 y_S5_C5 y_S5_C6 y_S5_C7 y_S5_C8 y_S5_C9 y_S5_C10 y_S5_C11 y_S5_C12 y_S6_C1 y_S6_C2 y_S6_C3 y_S6_C4 y_S6_C5 y_S6_C6 y_S6_C7 y_S6_C8 y_S6_C9 y_S6_C10 y_S6_C11 y_S6_C12
End
