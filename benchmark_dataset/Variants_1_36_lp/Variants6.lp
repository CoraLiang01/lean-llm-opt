Maximize
    obj: 28 x_F1_O1 + 34 x_F1_O2 + 30 x_F1_O3 + 24 x_F1_O4 + 25 x_F2_O1 + 31 x_F2_O2 + 36 x_F2_O3
      + 29 x_F2_O4 + 33 x_F3_O1 + 27 x_F3_O2 + 38 x_F3_O3 + 30 x_F3_O4 + 26 x_F4_O1 + 35 x_F4_O2
      + 32 x_F4_O3 + 28 x_F4_O4 + 30 x_F5_O1 + 37 x_F5_O2 + 29 x_F5_O3 + 34 x_F5_O4 + 24 x_F6_O1
      + 32 x_F6_O2 + 36 x_F6_O3 + 31 x_F6_O4
    Subject To
    choose_F1: x_F1_O1 + x_F1_O2 + x_F1_O3 + x_F1_O4 = 1
    choose_F2: x_F2_O1 + x_F2_O2 + x_F2_O3 + x_F2_O4 = 1
    choose_F3: x_F3_O1 + x_F3_O2 + x_F3_O3 + x_F3_O4 = 1
    choose_F4: x_F4_O1 + x_F4_O2 + x_F4_O3 + x_F4_O4 = 1
    choose_F5: x_F5_O1 + x_F5_O2 + x_F5_O3 + x_F5_O4 = 1
    choose_F6: x_F6_O1 + x_F6_O2 + x_F6_O3 + x_F6_O4 = 1
    weight_limit: 8 x_F1_O1 + 11 x_F1_O2 + 9 x_F1_O3 + 7 x_F1_O4 + 7 x_F2_O1 + 10 x_F2_O2
      + 12 x_F2_O3 + 9 x_F2_O4 + 10 x_F3_O1 + 8 x_F3_O2 + 13 x_F3_O3 + 9 x_F3_O4 + 6 x_F4_O1
      + 11 x_F4_O2 + 10 x_F4_O3 + 8 x_F4_O4 + 9 x_F5_O1 + 12 x_F5_O2 + 8 x_F5_O3 + 11 x_F5_O4
      + 7 x_F6_O1 + 10 x_F6_O2 + 12 x_F6_O3 + 9 x_F6_O4 <= 55
    budget_limit: 10 x_F1_O1 + 13 x_F1_O2 + 12 x_F1_O3 + 9 x_F1_O4 + 9 x_F2_O1 + 12 x_F2_O2
      + 14 x_F2_O3 + 11 x_F2_O4 + 12 x_F3_O1 + 10 x_F3_O2 + 15 x_F3_O3 + 11 x_F3_O4 + 8 x_F4_O1
      + 13 x_F4_O2 + 12 x_F4_O3 + 10 x_F4_O4 + 11 x_F5_O1 + 14 x_F5_O2 + 10 x_F5_O3 + 13 x_F5_O4
      + 9 x_F6_O1 + 12 x_F6_O2 + 14 x_F6_O3 + 11 x_F6_O4 <= 70
    Binaries
    x_F1_O1 x_F1_O2 x_F1_O3 x_F1_O4 x_F2_O1 x_F2_O2 x_F2_O3 x_F2_O4 x_F3_O1 x_F3_O2 x_F3_O3 x_F3_O4
    x_F4_O1 x_F4_O2 x_F4_O3 x_F4_O4 x_F5_O1 x_F5_O2 x_F5_O3 x_F5_O4 x_F6_O1 x_F6_O2 x_F6_O3 x_F6_O4
    End
