Minimize
    obj: 50 x_L1_A1 + 105 x_L1_A2 + 160 x_L1_A3 + 240 x_L1_A4 + 450 x_L1_A5 + 450 x_L1_A6
      + 260 x_L1_A7 + 770 x_L1_A8 + 900 x_L1_A9 + 360 x_L1_A10 + 462 x_L1_A11 + 380 x_L1_A12
      + 75 x_L2_A1 + 70 x_L2_A2 + 120 x_L2_A3 + 210 x_L2_A4 + 400 x_L2_A5 + 405 x_L2_A6
      + 240 x_L2_A7 + 715 x_L2_A8 + 840 x_L2_A9 + 330 x_L2_A10 + 420 x_L2_A11 + 342 x_L2_A12
      + 200 x_L3_A1 + 245 x_L3_A2 + 200 x_L3_A3 + 60 x_L3_A4 + 150 x_L3_A5 + 180 x_L3_A6
      + 160 x_L3_A7 + 495 x_L3_A8 + 660 x_L3_A9 + 210 x_L3_A10 + 252 x_L3_A11 + 266 x_L3_A12
      + 225 x_L4_A1 + 280 x_L4_A2 + 240 x_L4_A3 + 90 x_L4_A4 + 100 x_L4_A5 + 135 x_L4_A6
      + 140 x_L4_A7 + 440 x_L4_A8 + 600 x_L4_A9 + 180 x_L4_A10 + 210 x_L4_A11 + 228 x_L4_A12
      + 325 x_L5_A1 + 420 x_L5_A2 + 400 x_L5_A3 + 240 x_L5_A4 + 350 x_L5_A5 + 270 x_L5_A6
      + 40 x_L5_A7 + 165 x_L5_A8 + 240 x_L5_A9 + 150 x_L5_A10 + 252 x_L5_A11 + 266 x_L5_A12
      + 350 x_L6_A1 + 455 x_L6_A2 + 440 x_L6_A3 + 270 x_L6_A4 + 400 x_L6_A5 + 315 x_L6_A6
      + 60 x_L6_A7 + 110 x_L6_A8 + 180 x_L6_A9 + 120 x_L6_A10 + 210 x_L6_A11 + 228 x_L6_A12
      + 275 x_L7_A1 + 350 x_L7_A2 + 320 x_L7_A3 + 210 x_L7_A4 + 300 x_L7_A5 + 225 x_L7_A6
      + 120 x_L7_A7 + 275 x_L7_A8 + 240 x_L7_A9 + 60 x_L7_A10 + 126 x_L7_A11 + 76 x_L7_A12
    Subject To
    assign_A1: x_L1_A1 + x_L2_A1 + x_L3_A1 + x_L4_A1 + x_L5_A1 + x_L6_A1 + x_L7_A1 = 1
    assign_A2: x_L1_A2 + x_L2_A2 + x_L3_A2 + x_L4_A2 + x_L5_A2 + x_L6_A2 + x_L7_A2 = 1
    assign_A3: x_L1_A3 + x_L2_A3 + x_L3_A3 + x_L4_A3 + x_L5_A3 + x_L6_A3 + x_L7_A3 = 1
    assign_A4: x_L1_A4 + x_L2_A4 + x_L3_A4 + x_L4_A4 + x_L5_A4 + x_L6_A4 + x_L7_A4 = 1
    assign_A5: x_L1_A5 + x_L2_A5 + x_L3_A5 + x_L4_A5 + x_L5_A5 + x_L6_A5 + x_L7_A5 = 1
    assign_A6: x_L1_A6 + x_L2_A6 + x_L3_A6 + x_L4_A6 + x_L5_A6 + x_L6_A6 + x_L7_A6 = 1
    assign_A7: x_L1_A7 + x_L2_A7 + x_L3_A7 + x_L4_A7 + x_L5_A7 + x_L6_A7 + x_L7_A7 = 1
    assign_A8: x_L1_A8 + x_L2_A8 + x_L3_A8 + x_L4_A8 + x_L5_A8 + x_L6_A8 + x_L7_A8 = 1
    assign_A9: x_L1_A9 + x_L2_A9 + x_L3_A9 + x_L4_A9 + x_L5_A9 + x_L6_A9 + x_L7_A9 = 1
    assign_A10: x_L1_A10 + x_L2_A10 + x_L3_A10 + x_L4_A10 + x_L5_A10 + x_L6_A10 + x_L7_A10 = 1
    assign_A11: x_L1_A11 + x_L2_A11 + x_L3_A11 + x_L4_A11 + x_L5_A11 + x_L6_A11 + x_L7_A11 = 1
    assign_A12: x_L1_A12 + x_L2_A12 + x_L3_A12 + x_L4_A12 + x_L5_A12 + x_L6_A12 + x_L7_A12 = 1
    open_count: y_L1 + y_L2 + y_L3 + y_L4 + y_L5 + y_L6 + y_L7 = 3
    assign_open_link_L1_A1: x_L1_A1 - y_L1 <= 0
    assign_open_link_L1_A2: x_L1_A2 - y_L1 <= 0
    assign_open_link_L1_A3: x_L1_A3 - y_L1 <= 0
    assign_open_link_L1_A4: x_L1_A4 - y_L1 <= 0
    assign_open_link_L1_A5: x_L1_A5 - y_L1 <= 0
    assign_open_link_L1_A6: x_L1_A6 - y_L1 <= 0
    assign_open_link_L1_A7: x_L1_A7 - y_L1 <= 0
    assign_open_link_L1_A8: x_L1_A8 - y_L1 <= 0
    assign_open_link_L1_A9: x_L1_A9 - y_L1 <= 0
    assign_open_link_L1_A10: x_L1_A10 - y_L1 <= 0
    assign_open_link_L1_A11: x_L1_A11 - y_L1 <= 0
    assign_open_link_L1_A12: x_L1_A12 - y_L1 <= 0
    assign_open_link_L2_A1: x_L2_A1 - y_L2 <= 0
    assign_open_link_L2_A2: x_L2_A2 - y_L2 <= 0
    assign_open_link_L2_A3: x_L2_A3 - y_L2 <= 0
    assign_open_link_L2_A4: x_L2_A4 - y_L2 <= 0
    assign_open_link_L2_A5: x_L2_A5 - y_L2 <= 0
    assign_open_link_L2_A6: x_L2_A6 - y_L2 <= 0
    assign_open_link_L2_A7: x_L2_A7 - y_L2 <= 0
    assign_open_link_L2_A8: x_L2_A8 - y_L2 <= 0
    assign_open_link_L2_A9: x_L2_A9 - y_L2 <= 0
    assign_open_link_L2_A10: x_L2_A10 - y_L2 <= 0
    assign_open_link_L2_A11: x_L2_A11 - y_L2 <= 0
    assign_open_link_L2_A12: x_L2_A12 - y_L2 <= 0
    assign_open_link_L3_A1: x_L3_A1 - y_L3 <= 0
    assign_open_link_L3_A2: x_L3_A2 - y_L3 <= 0
    assign_open_link_L3_A3: x_L3_A3 - y_L3 <= 0
    assign_open_link_L3_A4: x_L3_A4 - y_L3 <= 0
    assign_open_link_L3_A5: x_L3_A5 - y_L3 <= 0
    assign_open_link_L3_A6: x_L3_A6 - y_L3 <= 0
    assign_open_link_L3_A7: x_L3_A7 - y_L3 <= 0
    assign_open_link_L3_A8: x_L3_A8 - y_L3 <= 0
    assign_open_link_L3_A9: x_L3_A9 - y_L3 <= 0
    assign_open_link_L3_A10: x_L3_A10 - y_L3 <= 0
    assign_open_link_L3_A11: x_L3_A11 - y_L3 <= 0
    assign_open_link_L3_A12: x_L3_A12 - y_L3 <= 0
    assign_open_link_L4_A1: x_L4_A1 - y_L4 <= 0
    assign_open_link_L4_A2: x_L4_A2 - y_L4 <= 0
    assign_open_link_L4_A3: x_L4_A3 - y_L4 <= 0
    assign_open_link_L4_A4: x_L4_A4 - y_L4 <= 0
    assign_open_link_L4_A5: x_L4_A5 - y_L4 <= 0
    assign_open_link_L4_A6: x_L4_A6 - y_L4 <= 0
    assign_open_link_L4_A7: x_L4_A7 - y_L4 <= 0
    assign_open_link_L4_A8: x_L4_A8 - y_L4 <= 0
    assign_open_link_L4_A9: x_L4_A9 - y_L4 <= 0
    assign_open_link_L4_A10: x_L4_A10 - y_L4 <= 0
    assign_open_link_L4_A11: x_L4_A11 - y_L4 <= 0
    assign_open_link_L4_A12: x_L4_A12 - y_L4 <= 0
    assign_open_link_L5_A1: x_L5_A1 - y_L5 <= 0
    assign_open_link_L5_A2: x_L5_A2 - y_L5 <= 0
    assign_open_link_L5_A3: x_L5_A3 - y_L5 <= 0
    assign_open_link_L5_A4: x_L5_A4 - y_L5 <= 0
    assign_open_link_L5_A5: x_L5_A5 - y_L5 <= 0
    assign_open_link_L5_A6: x_L5_A6 - y_L5 <= 0
    assign_open_link_L5_A7: x_L5_A7 - y_L5 <= 0
    assign_open_link_L5_A8: x_L5_A8 - y_L5 <= 0
    assign_open_link_L5_A9: x_L5_A9 - y_L5 <= 0
    assign_open_link_L5_A10: x_L5_A10 - y_L5 <= 0
    assign_open_link_L5_A11: x_L5_A11 - y_L5 <= 0
    assign_open_link_L5_A12: x_L5_A12 - y_L5 <= 0
    assign_open_link_L6_A1: x_L6_A1 - y_L6 <= 0
    assign_open_link_L6_A2: x_L6_A2 - y_L6 <= 0
    assign_open_link_L6_A3: x_L6_A3 - y_L6 <= 0
    assign_open_link_L6_A4: x_L6_A4 - y_L6 <= 0
    assign_open_link_L6_A5: x_L6_A5 - y_L6 <= 0
    assign_open_link_L6_A6: x_L6_A6 - y_L6 <= 0
    assign_open_link_L6_A7: x_L6_A7 - y_L6 <= 0
    assign_open_link_L6_A8: x_L6_A8 - y_L6 <= 0
    assign_open_link_L6_A9: x_L6_A9 - y_L6 <= 0
    assign_open_link_L6_A10: x_L6_A10 - y_L6 <= 0
    assign_open_link_L6_A11: x_L6_A11 - y_L6 <= 0
    assign_open_link_L6_A12: x_L6_A12 - y_L6 <= 0
    assign_open_link_L7_A1: x_L7_A1 - y_L7 <= 0
    assign_open_link_L7_A2: x_L7_A2 - y_L7 <= 0
    assign_open_link_L7_A3: x_L7_A3 - y_L7 <= 0
    assign_open_link_L7_A4: x_L7_A4 - y_L7 <= 0
    assign_open_link_L7_A5: x_L7_A5 - y_L7 <= 0
    assign_open_link_L7_A6: x_L7_A6 - y_L7 <= 0
    assign_open_link_L7_A7: x_L7_A7 - y_L7 <= 0
    assign_open_link_L7_A8: x_L7_A8 - y_L7 <= 0
    assign_open_link_L7_A9: x_L7_A9 - y_L7 <= 0
    assign_open_link_L7_A10: x_L7_A10 - y_L7 <= 0
    assign_open_link_L7_A11: x_L7_A11 - y_L7 <= 0
    assign_open_link_L7_A12: x_L7_A12 - y_L7 <= 0
    Binaries
    y_L1 y_L2 y_L3 y_L4 y_L5 y_L6 y_L7 x_L1_A1 x_L1_A2 x_L1_A3 x_L1_A4 x_L1_A5 x_L1_A6 x_L1_A7
    x_L1_A8 x_L1_A9 x_L1_A10 x_L1_A11 x_L1_A12 x_L2_A1 x_L2_A2 x_L2_A3 x_L2_A4 x_L2_A5 x_L2_A6
    x_L2_A7 x_L2_A8 x_L2_A9 x_L2_A10 x_L2_A11 x_L2_A12 x_L3_A1 x_L3_A2 x_L3_A3 x_L3_A4 x_L3_A5
    x_L3_A6 x_L3_A7 x_L3_A8 x_L3_A9 x_L3_A10 x_L3_A11 x_L3_A12 x_L4_A1 x_L4_A2 x_L4_A3 x_L4_A4
    x_L4_A5 x_L4_A6 x_L4_A7 x_L4_A8 x_L4_A9 x_L4_A10 x_L4_A11 x_L4_A12 x_L5_A1 x_L5_A2 x_L5_A3
    x_L5_A4 x_L5_A5 x_L5_A6 x_L5_A7 x_L5_A8 x_L5_A9 x_L5_A10 x_L5_A11 x_L5_A12 x_L6_A1 x_L6_A2
    x_L6_A3 x_L6_A4 x_L6_A5 x_L6_A6 x_L6_A7 x_L6_A8 x_L6_A9 x_L6_A10 x_L6_A11 x_L6_A12 x_L7_A1
    x_L7_A2 x_L7_A3 x_L7_A4 x_L7_A5 x_L7_A6 x_L7_A7 x_L7_A8 x_L7_A9 x_L7_A10 x_L7_A11 x_L7_A12
    End
