Minimize
    obj: 56 x_F1_N1 + 96 x_F1_N2 + 220 x_F1_N3 + 324 x_F1_N4 + 520 x_F1_N5 + 451 x_F1_N6
      + 325 x_F1_N7 + 672 x_F1_N8 + 825 x_F1_N9 + 360 x_F1_N10 + 112 x_F2_N1 + 64 x_F2_N2
      + 132 x_F2_N3 + 288 x_F2_N4 + 468 x_F2_N5 + 410 x_F2_N6 + 300 x_F2_N7 + 624 x_F2_N8
      + 770 x_F2_N9 + 330 x_F2_N10 + 252 x_F3_N1 + 256 x_F3_N2 + 176 x_F3_N3 + 72 x_F3_N4
      + 156 x_F3_N5 + 205 x_F3_N6 + 225 x_F3_N7 + 480 x_F3_N8 + 605 x_F3_N9 + 210 x_F3_N10
      + 280 x_F4_N1 + 288 x_F4_N2 + 264 x_F4_N3 + 108 x_F4_N4 + 104 x_F4_N5 + 123 x_F4_N6
      + 200 x_F4_N7 + 432 x_F4_N8 + 550 x_F4_N9 + 180 x_F4_N10 + 364 x_F5_N1 + 384 x_F5_N2
      + 440 x_F5_N3 + 288 x_F5_N4 + 364 x_F5_N5 + 246 x_F5_N6 + 50 x_F5_N7 + 144 x_F5_N8
      + 220 x_F5_N9 + 150 x_F5_N10 + 336 x_F6_N1 + 352 x_F6_N2 + 352 x_F6_N3 + 252 x_F6_N4
      + 312 x_F6_N5 + 205 x_F6_N6 + 125 x_F6_N7 + 192 x_F6_N8 + 165 x_F6_N9 + 60 x_F6_N10
    Subject To
    assign_N1: x_F1_N1 + x_F2_N1 + x_F3_N1 + x_F4_N1 + x_F5_N1 + x_F6_N1 = 1
    assign_N2: x_F1_N2 + x_F2_N2 + x_F3_N2 + x_F4_N2 + x_F5_N2 + x_F6_N2 = 1
    assign_N3: x_F1_N3 + x_F2_N3 + x_F3_N3 + x_F4_N3 + x_F5_N3 + x_F6_N3 = 1
    assign_N4: x_F1_N4 + x_F2_N4 + x_F3_N4 + x_F4_N4 + x_F5_N4 + x_F6_N4 = 1
    assign_N5: x_F1_N5 + x_F2_N5 + x_F3_N5 + x_F4_N5 + x_F5_N5 + x_F6_N5 = 1
    assign_N6: x_F1_N6 + x_F2_N6 + x_F3_N6 + x_F4_N6 + x_F5_N6 + x_F6_N6 = 1
    assign_N7: x_F1_N7 + x_F2_N7 + x_F3_N7 + x_F4_N7 + x_F5_N7 + x_F6_N7 = 1
    assign_N8: x_F1_N8 + x_F2_N8 + x_F3_N8 + x_F4_N8 + x_F5_N8 + x_F6_N8 = 1
    assign_N9: x_F1_N9 + x_F2_N9 + x_F3_N9 + x_F4_N9 + x_F5_N9 + x_F6_N9 = 1
    assign_N10: x_F1_N10 + x_F2_N10 + x_F3_N10 + x_F4_N10 + x_F5_N10 + x_F6_N10 = 1
    open_count: y_F1 + y_F2 + y_F3 + y_F4 + y_F5 + y_F6 = 2
    assign_open_link_F1_N1: x_F1_N1 - y_F1 <= 0
    assign_open_link_F1_N2: x_F1_N2 - y_F1 <= 0
    assign_open_link_F1_N3: x_F1_N3 - y_F1 <= 0
    assign_open_link_F1_N4: x_F1_N4 - y_F1 <= 0
    assign_open_link_F1_N5: x_F1_N5 - y_F1 <= 0
    assign_open_link_F1_N6: x_F1_N6 - y_F1 <= 0
    assign_open_link_F1_N7: x_F1_N7 - y_F1 <= 0
    assign_open_link_F1_N8: x_F1_N8 - y_F1 <= 0
    assign_open_link_F1_N9: x_F1_N9 - y_F1 <= 0
    assign_open_link_F1_N10: x_F1_N10 - y_F1 <= 0
    assign_open_link_F2_N1: x_F2_N1 - y_F2 <= 0
    assign_open_link_F2_N2: x_F2_N2 - y_F2 <= 0
    assign_open_link_F2_N3: x_F2_N3 - y_F2 <= 0
    assign_open_link_F2_N4: x_F2_N4 - y_F2 <= 0
    assign_open_link_F2_N5: x_F2_N5 - y_F2 <= 0
    assign_open_link_F2_N6: x_F2_N6 - y_F2 <= 0
    assign_open_link_F2_N7: x_F2_N7 - y_F2 <= 0
    assign_open_link_F2_N8: x_F2_N8 - y_F2 <= 0
    assign_open_link_F2_N9: x_F2_N9 - y_F2 <= 0
    assign_open_link_F2_N10: x_F2_N10 - y_F2 <= 0
    assign_open_link_F3_N1: x_F3_N1 - y_F3 <= 0
    assign_open_link_F3_N2: x_F3_N2 - y_F3 <= 0
    assign_open_link_F3_N3: x_F3_N3 - y_F3 <= 0
    assign_open_link_F3_N4: x_F3_N4 - y_F3 <= 0
    assign_open_link_F3_N5: x_F3_N5 - y_F3 <= 0
    assign_open_link_F3_N6: x_F3_N6 - y_F3 <= 0
    assign_open_link_F3_N7: x_F3_N7 - y_F3 <= 0
    assign_open_link_F3_N8: x_F3_N8 - y_F3 <= 0
    assign_open_link_F3_N9: x_F3_N9 - y_F3 <= 0
    assign_open_link_F3_N10: x_F3_N10 - y_F3 <= 0
    assign_open_link_F4_N1: x_F4_N1 - y_F4 <= 0
    assign_open_link_F4_N2: x_F4_N2 - y_F4 <= 0
    assign_open_link_F4_N3: x_F4_N3 - y_F4 <= 0
    assign_open_link_F4_N4: x_F4_N4 - y_F4 <= 0
    assign_open_link_F4_N5: x_F4_N5 - y_F4 <= 0
    assign_open_link_F4_N6: x_F4_N6 - y_F4 <= 0
    assign_open_link_F4_N7: x_F4_N7 - y_F4 <= 0
    assign_open_link_F4_N8: x_F4_N8 - y_F4 <= 0
    assign_open_link_F4_N9: x_F4_N9 - y_F4 <= 0
    assign_open_link_F4_N10: x_F4_N10 - y_F4 <= 0
    assign_open_link_F5_N1: x_F5_N1 - y_F5 <= 0
    assign_open_link_F5_N2: x_F5_N2 - y_F5 <= 0
    assign_open_link_F5_N3: x_F5_N3 - y_F5 <= 0
    assign_open_link_F5_N4: x_F5_N4 - y_F5 <= 0
    assign_open_link_F5_N5: x_F5_N5 - y_F5 <= 0
    assign_open_link_F5_N6: x_F5_N6 - y_F5 <= 0
    assign_open_link_F5_N7: x_F5_N7 - y_F5 <= 0
    assign_open_link_F5_N8: x_F5_N8 - y_F5 <= 0
    assign_open_link_F5_N9: x_F5_N9 - y_F5 <= 0
    assign_open_link_F5_N10: x_F5_N10 - y_F5 <= 0
    assign_open_link_F6_N1: x_F6_N1 - y_F6 <= 0
    assign_open_link_F6_N2: x_F6_N2 - y_F6 <= 0
    assign_open_link_F6_N3: x_F6_N3 - y_F6 <= 0
    assign_open_link_F6_N4: x_F6_N4 - y_F6 <= 0
    assign_open_link_F6_N5: x_F6_N5 - y_F6 <= 0
    assign_open_link_F6_N6: x_F6_N6 - y_F6 <= 0
    assign_open_link_F6_N7: x_F6_N7 - y_F6 <= 0
    assign_open_link_F6_N8: x_F6_N8 - y_F6 <= 0
    assign_open_link_F6_N9: x_F6_N9 - y_F6 <= 0
    assign_open_link_F6_N10: x_F6_N10 - y_F6 <= 0
    Binaries
    y_F1 y_F2 y_F3 y_F4 y_F5 y_F6 x_F1_N1 x_F1_N2 x_F1_N3 x_F1_N4 x_F1_N5 x_F1_N6 x_F1_N7 x_F1_N8
    x_F1_N9 x_F1_N10 x_F2_N1 x_F2_N2 x_F2_N3 x_F2_N4 x_F2_N5 x_F2_N6 x_F2_N7 x_F2_N8 x_F2_N9
    x_F2_N10 x_F3_N1 x_F3_N2 x_F3_N3 x_F3_N4 x_F3_N5 x_F3_N6 x_F3_N7 x_F3_N8 x_F3_N9 x_F3_N10
    x_F4_N1 x_F4_N2 x_F4_N3 x_F4_N4 x_F4_N5 x_F4_N6 x_F4_N7 x_F4_N8 x_F4_N9 x_F4_N10 x_F5_N1 x_F5_N2
    x_F5_N3 x_F5_N4 x_F5_N5 x_F5_N6 x_F5_N7 x_F5_N8 x_F5_N9 x_F5_N10 x_F6_N1 x_F6_N2 x_F6_N3 x_F6_N4
    x_F6_N5 x_F6_N6 x_F6_N7 x_F6_N8 x_F6_N9 x_F6_N10
    End
