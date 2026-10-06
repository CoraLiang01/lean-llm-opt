Minimize
    obj: 6 y_V1_V2 + 4 y_V1_V3 + 9 y_V1_V4 + 5 y_V2_V3 + 7 y_V2_V5 + 3 y_V3_V4 + 8 y_V3_V6
      + 4 y_V4_V6 + 10 y_V4_V7 + 6 y_V5_V6 + 11 y_V5_V8 + 5 y_V6_V7 + 7 y_V6_V8 + 4 y_V7_V8
      + 6 y_V7_V9 + 5 y_V8_V9
    Subject To
    edge_count: y_V1_V2 + y_V1_V3 + y_V1_V4 + y_V2_V3 + y_V2_V5 + y_V3_V4 + y_V3_V6 + y_V4_V6
      + y_V4_V7 + y_V5_V6 + y_V5_V8 + y_V6_V7 + y_V6_V8 + y_V7_V8 + y_V7_V9 + y_V8_V9 = 8
    flow_balance_V1: f_V1_V2 - f_V2_V1 + f_V1_V3 - f_V3_V1 + f_V1_V4 - f_V4_V1 = 8
    flow_balance_V2: f_V1_V2 - f_V2_V1 - f_V2_V3 + f_V3_V2 - f_V2_V5 + f_V5_V2 = 1
    flow_balance_V3: f_V1_V3 - f_V3_V1 + f_V2_V3 - f_V3_V2 - f_V3_V4 + f_V4_V3 - f_V3_V6 + f_V6_V3
      = 1
    flow_balance_V4: f_V1_V4 - f_V4_V1 + f_V3_V4 - f_V4_V3 - f_V4_V6 + f_V6_V4 - f_V4_V7 + f_V7_V4
      = 1
    flow_balance_V5: f_V2_V5 - f_V5_V2 - f_V5_V6 + f_V6_V5 - f_V5_V8 + f_V8_V5 = 1
    flow_balance_V6: f_V3_V6 - f_V6_V3 + f_V4_V6 - f_V6_V4 + f_V5_V6 - f_V6_V5 - f_V6_V7 + f_V7_V6
      - f_V6_V8 + f_V8_V6 = 1
    flow_balance_V7: f_V4_V7 - f_V7_V4 + f_V6_V7 - f_V7_V6 - f_V7_V8 + f_V8_V7 - f_V7_V9 + f_V9_V7
      = 1
    flow_balance_V8: f_V5_V8 - f_V8_V5 + f_V6_V8 - f_V8_V6 + f_V7_V8 - f_V8_V7 - f_V8_V9 + f_V9_V8
      = 1
    flow_balance_V9: f_V7_V9 - f_V9_V7 + f_V8_V9 - f_V9_V8 = 1
    flow_link_V1_V2: f_V1_V2 - 8 y_V1_V2 <= 0
    flow_link_V2_V1: f_V2_V1 - 8 y_V1_V2 <= 0
    flow_link_V1_V3: f_V1_V3 - 8 y_V1_V3 <= 0
    flow_link_V3_V1: f_V3_V1 - 8 y_V1_V3 <= 0
    flow_link_V1_V4: f_V1_V4 - 8 y_V1_V4 <= 0
    flow_link_V4_V1: f_V4_V1 - 8 y_V1_V4 <= 0
    flow_link_V2_V3: f_V2_V3 - 8 y_V2_V3 <= 0
    flow_link_V3_V2: f_V3_V2 - 8 y_V2_V3 <= 0
    flow_link_V2_V5: f_V2_V5 - 8 y_V2_V5 <= 0
    flow_link_V5_V2: f_V5_V2 - 8 y_V2_V5 <= 0
    flow_link_V3_V4: f_V3_V4 - 8 y_V3_V4 <= 0
    flow_link_V4_V3: f_V4_V3 - 8 y_V3_V4 <= 0
    flow_link_V3_V6: f_V3_V6 - 8 y_V3_V6 <= 0
    flow_link_V6_V3: f_V6_V3 - 8 y_V3_V6 <= 0
    flow_link_V4_V6: f_V4_V6 - 8 y_V4_V6 <= 0
    flow_link_V6_V4: f_V6_V4 - 8 y_V4_V6 <= 0
    flow_link_V4_V7: f_V4_V7 - 8 y_V4_V7 <= 0
    flow_link_V7_V4: f_V7_V4 - 8 y_V4_V7 <= 0
    flow_link_V5_V6: f_V5_V6 - 8 y_V5_V6 <= 0
    flow_link_V6_V5: f_V6_V5 - 8 y_V5_V6 <= 0
    flow_link_V5_V8: f_V5_V8 - 8 y_V5_V8 <= 0
    flow_link_V8_V5: f_V8_V5 - 8 y_V5_V8 <= 0
    flow_link_V6_V7: f_V6_V7 - 8 y_V6_V7 <= 0
    flow_link_V7_V6: f_V7_V6 - 8 y_V6_V7 <= 0
    flow_link_V6_V8: f_V6_V8 - 8 y_V6_V8 <= 0
    flow_link_V8_V6: f_V8_V6 - 8 y_V6_V8 <= 0
    flow_link_V7_V8: f_V7_V8 - 8 y_V7_V8 <= 0
    flow_link_V8_V7: f_V8_V7 - 8 y_V7_V8 <= 0
    flow_link_V7_V9: f_V7_V9 - 8 y_V7_V9 <= 0
    flow_link_V9_V7: f_V9_V7 - 8 y_V7_V9 <= 0
    flow_link_V8_V9: f_V8_V9 - 8 y_V8_V9 <= 0
    flow_link_V9_V8: f_V9_V8 - 8 y_V8_V9 <= 0
    Bounds
    f_V1_V2 >= 0
    f_V2_V1 >= 0
    f_V1_V3 >= 0
    f_V3_V1 >= 0
    f_V1_V4 >= 0
    f_V4_V1 >= 0
    f_V2_V3 >= 0
    f_V3_V2 >= 0
    f_V2_V5 >= 0
    f_V5_V2 >= 0
    f_V3_V4 >= 0
    f_V4_V3 >= 0
    f_V3_V6 >= 0
    f_V6_V3 >= 0
    f_V4_V6 >= 0
    f_V6_V4 >= 0
    f_V4_V7 >= 0
    f_V7_V4 >= 0
    f_V5_V6 >= 0
    f_V6_V5 >= 0
    f_V5_V8 >= 0
    f_V8_V5 >= 0
    f_V6_V7 >= 0
    f_V7_V6 >= 0
    f_V6_V8 >= 0
    f_V8_V6 >= 0
    f_V7_V8 >= 0
    f_V8_V7 >= 0
    f_V7_V9 >= 0
    f_V9_V7 >= 0
    f_V8_V9 >= 0
    f_V9_V8 >= 0
    Binaries
    y_V1_V2 y_V1_V3 y_V1_V4 y_V2_V3 y_V2_V5 y_V3_V4 y_V3_V6 y_V4_V6 y_V4_V7 y_V5_V6 y_V5_V8 y_V6_V7
    y_V6_V8 y_V7_V8 y_V7_V9 y_V8_V9
    End
