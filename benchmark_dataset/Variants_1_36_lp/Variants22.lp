Maximize
    obj: 18 x_C1_O1 + 27 x_C1_O2 + 30 x_C1_O3 + 24 x_C2_O1 + 32 x_C2_O2 + 28 x_C2_O3 + 22 x_C3_O1
      + 35 x_C3_O2 + 31 x_C3_O3 + 20 x_C4_O1 + 29 x_C4_O2 + 34 x_C4_O3 + 23 x_C5_O1 + 33 x_C5_O2
      + 26 x_C5_O3 + 21 x_C6_O1 + 30 x_C6_O2 + 36 x_C6_O3
    Subject To
    choose_C1: x_C1_O1 + x_C1_O2 + x_C1_O3 = 1
    choose_C2: x_C2_O1 + x_C2_O2 + x_C2_O3 = 1
    choose_C3: x_C3_O1 + x_C3_O2 + x_C3_O3 = 1
    choose_C4: x_C4_O1 + x_C4_O2 + x_C4_O3 = 1
    choose_C5: x_C5_O1 + x_C5_O2 + x_C5_O3 = 1
    choose_C6: x_C6_O1 + x_C6_O2 + x_C6_O3 = 1
    weight_limit: 5 x_C1_O1 + 8 x_C1_O2 + 10 x_C1_O3 + 7 x_C2_O1 + 11 x_C2_O2 + 9 x_C2_O3
      + 6 x_C3_O1 + 12 x_C3_O2 + 10 x_C3_O3 + 5 x_C4_O1 + 9 x_C4_O2 + 12 x_C4_O3 + 7 x_C5_O1
      + 11 x_C5_O2 + 8 x_C5_O3 + 6 x_C6_O1 + 10 x_C6_O2 + 13 x_C6_O3 <= 55
    labor_limit: 6 x_C1_O1 + 9 x_C1_O2 + 11 x_C1_O3 + 8 x_C2_O1 + 13 x_C2_O2 + 10 x_C2_O3
      + 7 x_C3_O1 + 14 x_C3_O2 + 12 x_C3_O3 + 8 x_C4_O1 + 11 x_C4_O2 + 13 x_C4_O3 + 7 x_C5_O1
      + 12 x_C5_O2 + 10 x_C5_O3 + 6 x_C6_O1 + 11 x_C6_O2 + 15 x_C6_O3 <= 64
    Binaries
    x_C1_O1 x_C1_O2 x_C1_O3 x_C2_O1 x_C2_O2 x_C2_O3 x_C3_O1 x_C3_O2 x_C3_O3 x_C4_O1 x_C4_O2 x_C4_O3
    x_C5_O1 x_C5_O2 x_C5_O3 x_C6_O1 x_C6_O2 x_C6_O3
    End
