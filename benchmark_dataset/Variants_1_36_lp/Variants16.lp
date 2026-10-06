Maximize
    obj: 22 x_G1_O1 + 29 x_G1_O2 + 31 x_G1_O3 + 25 x_G1_O4 + 24 x_G2_O1 + 33 x_G2_O2 + 28 x_G2_O3
      + 35 x_G2_O4 + 30 x_G3_O1 + 26 x_G3_O2 + 38 x_G3_O3 + 34 x_G3_O4 + 21 x_G4_O1 + 32 x_G4_O2
      + 36 x_G4_O3 + 27 x_G4_O4 + 29 x_G5_O1 + 37 x_G5_O2 + 33 x_G5_O3 + 24 x_G5_O4
    Subject To
    choose_G1: x_G1_O1 + x_G1_O2 + x_G1_O3 + x_G1_O4 = 1
    choose_G2: x_G2_O1 + x_G2_O2 + x_G2_O3 + x_G2_O4 = 1
    choose_G3: x_G3_O1 + x_G3_O2 + x_G3_O3 + x_G3_O4 = 1
    choose_G4: x_G4_O1 + x_G4_O2 + x_G4_O3 + x_G4_O4 = 1
    choose_G5: x_G5_O1 + x_G5_O2 + x_G5_O3 + x_G5_O4 = 1
    weight_limit: 6 x_G1_O1 + 9 x_G1_O2 + 10 x_G1_O3 + 7 x_G1_O4 + 7 x_G2_O1 + 11 x_G2_O2
      + 8 x_G2_O3 + 12 x_G2_O4 + 9 x_G3_O1 + 7 x_G3_O2 + 13 x_G3_O3 + 11 x_G3_O4 + 5 x_G4_O1
      + 10 x_G4_O2 + 12 x_G4_O3 + 8 x_G4_O4 + 8 x_G5_O1 + 12 x_G5_O2 + 10 x_G5_O3 + 6 x_G5_O4 <= 48
    budget_limit: 8 x_G1_O1 + 11 x_G1_O2 + 13 x_G1_O3 + 9 x_G1_O4 + 8 x_G2_O1 + 13 x_G2_O2
      + 10 x_G2_O3 + 14 x_G2_O4 + 12 x_G3_O1 + 9 x_G3_O2 + 16 x_G3_O3 + 13 x_G3_O4 + 7 x_G4_O1
      + 12 x_G4_O2 + 15 x_G4_O3 + 10 x_G4_O4 + 11 x_G5_O1 + 15 x_G5_O2 + 12 x_G5_O3 + 8 x_G5_O4
      <= 60
    Binaries
    x_G1_O1 x_G1_O2 x_G1_O3 x_G1_O4 x_G2_O1 x_G2_O2 x_G2_O3 x_G2_O4 x_G3_O1 x_G3_O2 x_G3_O3 x_G3_O4
    x_G4_O1 x_G4_O2 x_G4_O3 x_G4_O4 x_G5_O1 x_G5_O2 x_G5_O3 x_G5_O4
    End
