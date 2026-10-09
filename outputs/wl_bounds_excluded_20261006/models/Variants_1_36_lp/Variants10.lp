\ Canonical native-bound formulation; bounds excluded from WL graph.
Minimize
  obj: + 4 y_N1_N2 + 3 y_N1_N3 + 9 y_N1_N4 + 5 y_N2_N3 + 6 y_N2_N5 + 4 y_N3_N4 + 7 y_N3_N6 + 2 y_N4_N6 + 8 y_N4_N7 + 3 y_N5_N6 + 10 y_N5_N8 + 4 y_N6_N7 + 6 y_N6_N8 + 5 y_N7_N8
Subject To
  edge_count: + 1 y_N1_N2 + 1 y_N1_N3 + 1 y_N1_N4 + 1 y_N2_N3 + 1 y_N2_N5 + 1 y_N3_N4 + 1 y_N3_N6 + 1 y_N4_N6 + 1 y_N4_N7 + 1 y_N5_N6 + 1 y_N5_N8 + 1 y_N6_N7 + 1 y_N6_N8 + 1 y_N7_N8 = 7
  flow_balance_N1: + 1 f_N1_N2 - 1 f_N2_N1 + 1 f_N1_N3 - 1 f_N3_N1 + 1 f_N1_N4 - 1 f_N4_N1 = 7
  flow_balance_N2: + 1 f_N1_N2 - 1 f_N2_N1 - 1 f_N2_N3 + 1 f_N3_N2 - 1 f_N2_N5 + 1 f_N5_N2 = 1
  flow_balance_N3: + 1 f_N1_N3 - 1 f_N3_N1 + 1 f_N2_N3 - 1 f_N3_N2 - 1 f_N3_N4 + 1 f_N4_N3 - 1 f_N3_N6 + 1 f_N6_N3 = 1
  flow_balance_N4: + 1 f_N1_N4 - 1 f_N4_N1 + 1 f_N3_N4 - 1 f_N4_N3 - 1 f_N4_N6 + 1 f_N6_N4 - 1 f_N4_N7 + 1 f_N7_N4 = 1
  flow_balance_N5: + 1 f_N2_N5 - 1 f_N5_N2 - 1 f_N5_N6 + 1 f_N6_N5 - 1 f_N5_N8 + 1 f_N8_N5 = 1
  flow_balance_N6: + 1 f_N3_N6 - 1 f_N6_N3 + 1 f_N4_N6 - 1 f_N6_N4 + 1 f_N5_N6 - 1 f_N6_N5 - 1 f_N6_N7 + 1 f_N7_N6 - 1 f_N6_N8 + 1 f_N8_N6 = 1
  flow_balance_N7: + 1 f_N4_N7 - 1 f_N7_N4 + 1 f_N6_N7 - 1 f_N7_N6 - 1 f_N7_N8 + 1 f_N8_N7 = 1
  flow_balance_N8: + 1 f_N5_N8 - 1 f_N8_N5 + 1 f_N6_N8 - 1 f_N8_N6 + 1 f_N7_N8 - 1 f_N8_N7 = 1
  flow_link_N1_N2: - 7 y_N1_N2 + 1 f_N1_N2 <= 0
  flow_link_N2_N1: - 7 y_N1_N2 + 1 f_N2_N1 <= 0
  flow_link_N1_N3: - 7 y_N1_N3 + 1 f_N1_N3 <= 0
  flow_link_N3_N1: - 7 y_N1_N3 + 1 f_N3_N1 <= 0
  flow_link_N1_N4: - 7 y_N1_N4 + 1 f_N1_N4 <= 0
  flow_link_N4_N1: - 7 y_N1_N4 + 1 f_N4_N1 <= 0
  flow_link_N2_N3: - 7 y_N2_N3 + 1 f_N2_N3 <= 0
  flow_link_N3_N2: - 7 y_N2_N3 + 1 f_N3_N2 <= 0
  flow_link_N2_N5: - 7 y_N2_N5 + 1 f_N2_N5 <= 0
  flow_link_N5_N2: - 7 y_N2_N5 + 1 f_N5_N2 <= 0
  flow_link_N3_N4: - 7 y_N3_N4 + 1 f_N3_N4 <= 0
  flow_link_N4_N3: - 7 y_N3_N4 + 1 f_N4_N3 <= 0
  flow_link_N3_N6: - 7 y_N3_N6 + 1 f_N3_N6 <= 0
  flow_link_N6_N3: - 7 y_N3_N6 + 1 f_N6_N3 <= 0
  flow_link_N4_N6: - 7 y_N4_N6 + 1 f_N4_N6 <= 0
  flow_link_N6_N4: - 7 y_N4_N6 + 1 f_N6_N4 <= 0
  flow_link_N4_N7: - 7 y_N4_N7 + 1 f_N4_N7 <= 0
  flow_link_N7_N4: - 7 y_N4_N7 + 1 f_N7_N4 <= 0
  flow_link_N5_N6: - 7 y_N5_N6 + 1 f_N5_N6 <= 0
  flow_link_N6_N5: - 7 y_N5_N6 + 1 f_N6_N5 <= 0
  flow_link_N5_N8: - 7 y_N5_N8 + 1 f_N5_N8 <= 0
  flow_link_N8_N5: - 7 y_N5_N8 + 1 f_N8_N5 <= 0
  flow_link_N6_N7: - 7 y_N6_N7 + 1 f_N6_N7 <= 0
  flow_link_N7_N6: - 7 y_N6_N7 + 1 f_N7_N6 <= 0
  flow_link_N6_N8: - 7 y_N6_N8 + 1 f_N6_N8 <= 0
  flow_link_N8_N6: - 7 y_N6_N8 + 1 f_N8_N6 <= 0
  flow_link_N7_N8: - 7 y_N7_N8 + 1 f_N7_N8 <= 0
  flow_link_N8_N7: - 7 y_N7_N8 + 1 f_N8_N7 <= 0
Bounds
  0 <= y_N1_N2 <= 1
  0 <= y_N1_N3 <= 1
  0 <= y_N1_N4 <= 1
  0 <= y_N2_N3 <= 1
  0 <= y_N2_N5 <= 1
  0 <= y_N3_N4 <= 1
  0 <= y_N3_N6 <= 1
  0 <= y_N4_N6 <= 1
  0 <= y_N4_N7 <= 1
  0 <= y_N5_N6 <= 1
  0 <= y_N5_N8 <= 1
  0 <= y_N6_N7 <= 1
  0 <= y_N6_N8 <= 1
  0 <= y_N7_N8 <= 1
  0 <= f_N1_N2 <= +inf
  0 <= f_N2_N1 <= +inf
  0 <= f_N1_N3 <= +inf
  0 <= f_N3_N1 <= +inf
  0 <= f_N1_N4 <= +inf
  0 <= f_N4_N1 <= +inf
  0 <= f_N2_N3 <= +inf
  0 <= f_N3_N2 <= +inf
  0 <= f_N2_N5 <= +inf
  0 <= f_N5_N2 <= +inf
  0 <= f_N3_N4 <= +inf
  0 <= f_N4_N3 <= +inf
  0 <= f_N3_N6 <= +inf
  0 <= f_N6_N3 <= +inf
  0 <= f_N4_N6 <= +inf
  0 <= f_N6_N4 <= +inf
  0 <= f_N4_N7 <= +inf
  0 <= f_N7_N4 <= +inf
  0 <= f_N5_N6 <= +inf
  0 <= f_N6_N5 <= +inf
  0 <= f_N5_N8 <= +inf
  0 <= f_N8_N5 <= +inf
  0 <= f_N6_N7 <= +inf
  0 <= f_N7_N6 <= +inf
  0 <= f_N6_N8 <= +inf
  0 <= f_N8_N6 <= +inf
  0 <= f_N7_N8 <= +inf
  0 <= f_N8_N7 <= +inf
Binaries
  y_N1_N2 y_N1_N3 y_N1_N4 y_N2_N3 y_N2_N5 y_N3_N4 y_N3_N6 y_N4_N6 y_N4_N7 y_N5_N6 y_N5_N8 y_N6_N7 y_N6_N8 y_N7_N8
End
