Minimize
    obj: 4 y_N1_N2 + 3 y_N1_N3 + 9 y_N1_N4 + 5 y_N2_N3 + 6 y_N2_N5 + 4 y_N3_N4 + 7 y_N3_N6
      + 2 y_N4_N6 + 8 y_N4_N7 + 3 y_N5_N6 + 10 y_N5_N8 + 4 y_N6_N7 + 6 y_N6_N8 + 5 y_N7_N8
    Subject To
    edge_count: y_N1_N2 + y_N1_N3 + y_N1_N4 + y_N2_N3 + y_N2_N5 + y_N3_N4 + y_N3_N6 + y_N4_N6
      + y_N4_N7 + y_N5_N6 + y_N5_N8 + y_N6_N7 + y_N6_N8 + y_N7_N8 = 7
    flow_balance_N1: f_N1_N2 - f_N2_N1 + f_N1_N3 - f_N3_N1 + f_N1_N4 - f_N4_N1 = 7
    flow_balance_N2: f_N1_N2 - f_N2_N1 - f_N2_N3 + f_N3_N2 - f_N2_N5 + f_N5_N2 = 1
    flow_balance_N3: f_N1_N3 - f_N3_N1 + f_N2_N3 - f_N3_N2 - f_N3_N4 + f_N4_N3 - f_N3_N6 + f_N6_N3
      = 1
    flow_balance_N4: f_N1_N4 - f_N4_N1 + f_N3_N4 - f_N4_N3 - f_N4_N6 + f_N6_N4 - f_N4_N7 + f_N7_N4
      = 1
    flow_balance_N5: f_N2_N5 - f_N5_N2 - f_N5_N6 + f_N6_N5 - f_N5_N8 + f_N8_N5 = 1
    flow_balance_N6: f_N3_N6 - f_N6_N3 + f_N4_N6 - f_N6_N4 + f_N5_N6 - f_N6_N5 - f_N6_N7 + f_N7_N6
      - f_N6_N8 + f_N8_N6 = 1
    flow_balance_N7: f_N4_N7 - f_N7_N4 + f_N6_N7 - f_N7_N6 - f_N7_N8 + f_N8_N7 = 1
    flow_balance_N8: f_N5_N8 - f_N8_N5 + f_N6_N8 - f_N8_N6 + f_N7_N8 - f_N8_N7 = 1
    flow_link_N1_N2: f_N1_N2 - 7 y_N1_N2 <= 0
    flow_link_N2_N1: f_N2_N1 - 7 y_N1_N2 <= 0
    flow_link_N1_N3: f_N1_N3 - 7 y_N1_N3 <= 0
    flow_link_N3_N1: f_N3_N1 - 7 y_N1_N3 <= 0
    flow_link_N1_N4: f_N1_N4 - 7 y_N1_N4 <= 0
    flow_link_N4_N1: f_N4_N1 - 7 y_N1_N4 <= 0
    flow_link_N2_N3: f_N2_N3 - 7 y_N2_N3 <= 0
    flow_link_N3_N2: f_N3_N2 - 7 y_N2_N3 <= 0
    flow_link_N2_N5: f_N2_N5 - 7 y_N2_N5 <= 0
    flow_link_N5_N2: f_N5_N2 - 7 y_N2_N5 <= 0
    flow_link_N3_N4: f_N3_N4 - 7 y_N3_N4 <= 0
    flow_link_N4_N3: f_N4_N3 - 7 y_N3_N4 <= 0
    flow_link_N3_N6: f_N3_N6 - 7 y_N3_N6 <= 0
    flow_link_N6_N3: f_N6_N3 - 7 y_N3_N6 <= 0
    flow_link_N4_N6: f_N4_N6 - 7 y_N4_N6 <= 0
    flow_link_N6_N4: f_N6_N4 - 7 y_N4_N6 <= 0
    flow_link_N4_N7: f_N4_N7 - 7 y_N4_N7 <= 0
    flow_link_N7_N4: f_N7_N4 - 7 y_N4_N7 <= 0
    flow_link_N5_N6: f_N5_N6 - 7 y_N5_N6 <= 0
    flow_link_N6_N5: f_N6_N5 - 7 y_N5_N6 <= 0
    flow_link_N5_N8: f_N5_N8 - 7 y_N5_N8 <= 0
    flow_link_N8_N5: f_N8_N5 - 7 y_N5_N8 <= 0
    flow_link_N6_N7: f_N6_N7 - 7 y_N6_N7 <= 0
    flow_link_N7_N6: f_N7_N6 - 7 y_N6_N7 <= 0
    flow_link_N6_N8: f_N6_N8 - 7 y_N6_N8 <= 0
    flow_link_N8_N6: f_N8_N6 - 7 y_N6_N8 <= 0
    flow_link_N7_N8: f_N7_N8 - 7 y_N7_N8 <= 0
    flow_link_N8_N7: f_N8_N7 - 7 y_N7_N8 <= 0
    Bounds
    f_N1_N2 >= 0
    f_N2_N1 >= 0
    f_N1_N3 >= 0
    f_N3_N1 >= 0
    f_N1_N4 >= 0
    f_N4_N1 >= 0
    f_N2_N3 >= 0
    f_N3_N2 >= 0
    f_N2_N5 >= 0
    f_N5_N2 >= 0
    f_N3_N4 >= 0
    f_N4_N3 >= 0
    f_N3_N6 >= 0
    f_N6_N3 >= 0
    f_N4_N6 >= 0
    f_N6_N4 >= 0
    f_N4_N7 >= 0
    f_N7_N4 >= 0
    f_N5_N6 >= 0
    f_N6_N5 >= 0
    f_N5_N8 >= 0
    f_N8_N5 >= 0
    f_N6_N7 >= 0
    f_N7_N6 >= 0
    f_N6_N8 >= 0
    f_N8_N6 >= 0
    f_N7_N8 >= 0
    f_N8_N7 >= 0
    Binaries
    y_N1_N2 y_N1_N3 y_N1_N4 y_N2_N3 y_N2_N5 y_N3_N4 y_N3_N6 y_N4_N6 y_N4_N7 y_N5_N6 y_N5_N8 y_N6_N7
    y_N6_N8 y_N7_N8
    End
