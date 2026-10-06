Minimize
    obj: 8 x_M1_J1 + 7 x_M1_J2 + 25 x_M1_J3 + 24 x_M1_J4 + 27 x_M1_J5 + 26 x_M1_J6 + 28 x_M1_J7
      + 29 x_M1_J8 + 23 x_M2_J1 + 24 x_M2_J2 + 6 x_M2_J3 + 9 x_M2_J4 + 25 x_M2_J5 + 27 x_M2_J6
      + 26 x_M2_J7 + 28 x_M2_J8 + 27 x_M3_J1 + 26 x_M3_J2 + 24 x_M3_J3 + 25 x_M3_J4 + 5 x_M3_J5
      + 8 x_M3_J6 + 23 x_M3_J7 + 24 x_M3_J8 + 25 x_M4_J1 + 27 x_M4_J2 + 26 x_M4_J3 + 24 x_M4_J4
      + 23 x_M4_J5 + 25 x_M4_J6 + 6 x_M4_J7 + 7 x_M4_J8
    Subject To
    assign_J1: x_M1_J1 + x_M2_J1 + x_M3_J1 + x_M4_J1 = 1
    assign_J2: x_M1_J2 + x_M2_J2 + x_M3_J2 + x_M4_J2 = 1
    assign_J3: x_M1_J3 + x_M2_J3 + x_M3_J3 + x_M4_J3 = 1
    assign_J4: x_M1_J4 + x_M2_J4 + x_M3_J4 + x_M4_J4 = 1
    assign_J5: x_M1_J5 + x_M2_J5 + x_M3_J5 + x_M4_J5 = 1
    assign_J6: x_M1_J6 + x_M2_J6 + x_M3_J6 + x_M4_J6 = 1
    assign_J7: x_M1_J7 + x_M2_J7 + x_M3_J7 + x_M4_J7 = 1
    assign_J8: x_M1_J8 + x_M2_J8 + x_M3_J8 + x_M4_J8 = 1
    capacity_M1: 5 x_M1_J1 + 6 x_M1_J2 + 8 x_M1_J3 + 7 x_M1_J4 + 9 x_M1_J5 + 8 x_M1_J6 + 7 x_M1_J7
      + 7 x_M1_J8 <= 13
    capacity_M2: 8 x_M2_J1 + 7 x_M2_J2 + 4 x_M2_J3 + 7 x_M2_J4 + 8 x_M2_J5 + 9 x_M2_J6 + 8 x_M2_J7
      + 7 x_M2_J8 <= 12
    capacity_M3: 9 x_M3_J1 + 8 x_M3_J2 + 7 x_M3_J3 + 8 x_M3_J4 + 6 x_M3_J5 + 5 x_M3_J6 + 8 x_M3_J7
      + 7 x_M3_J8 <= 12
    capacity_M4: 8 x_M4_J1 + 8 x_M4_J2 + 7 x_M4_J3 + 8 x_M4_J4 + 8 x_M4_J5 + 7 x_M4_J6 + 5 x_M4_J7
      + 6 x_M4_J8 <= 12
    Binaries
    x_M1_J1 x_M1_J2 x_M1_J3 x_M1_J4 x_M1_J5 x_M1_J6 x_M1_J7 x_M1_J8 x_M2_J1 x_M2_J2 x_M2_J3 x_M2_J4
    x_M2_J5 x_M2_J6 x_M2_J7 x_M2_J8 x_M3_J1 x_M3_J2 x_M3_J3 x_M3_J4 x_M3_J5 x_M3_J6 x_M3_J7 x_M3_J8
    x_M4_J1 x_M4_J2 x_M4_J3 x_M4_J4 x_M4_J5 x_M4_J6 x_M4_J7 x_M4_J8
    End
