\ Canonical native-bound formulation; bounds excluded from WL graph.
Minimize
  obj: + 6 x_W1_J1 + 8 x_W1_J2 + 18 x_W1_J3 + 20 x_W1_J4 + 21 x_W1_J5 + 19 x_W1_J6 + 23 x_W1_J7 + 22 x_W1_J8 + 24 x_W1_J9 + 19 x_W2_J1 + 18 x_W2_J2 + 7 x_W2_J3 + 6 x_W2_J4 + 20 x_W2_J5 + 22 x_W2_J6 + 21 x_W2_J7 + 23 x_W2_J8 + 25 x_W2_J9 + 22 x_W3_J1 + 21 x_W3_J2 + 20 x_W3_J3 + 19 x_W3_J4 + 5 x_W3_J5 + 7 x_W3_J6 + 18 x_W3_J7 + 20 x_W3_J8 + 21 x_W3_J9 + 21 x_W4_J1 + 22 x_W4_J2 + 23 x_W4_J3 + 20 x_W4_J4 + 19 x_W4_J5 + 18 x_W4_J6 + 6 x_W4_J7 + 8 x_W4_J8 + 7 x_W4_J9
Subject To
  assign_J1: + 1 x_W1_J1 + 1 x_W2_J1 + 1 x_W3_J1 + 1 x_W4_J1 = 1
  assign_J2: + 1 x_W1_J2 + 1 x_W2_J2 + 1 x_W3_J2 + 1 x_W4_J2 = 1
  assign_J3: + 1 x_W1_J3 + 1 x_W2_J3 + 1 x_W3_J3 + 1 x_W4_J3 = 1
  assign_J4: + 1 x_W1_J4 + 1 x_W2_J4 + 1 x_W3_J4 + 1 x_W4_J4 = 1
  assign_J5: + 1 x_W1_J5 + 1 x_W2_J5 + 1 x_W3_J5 + 1 x_W4_J5 = 1
  assign_J6: + 1 x_W1_J6 + 1 x_W2_J6 + 1 x_W3_J6 + 1 x_W4_J6 = 1
  assign_J7: + 1 x_W1_J7 + 1 x_W2_J7 + 1 x_W3_J7 + 1 x_W4_J7 = 1
  assign_J8: + 1 x_W1_J8 + 1 x_W2_J8 + 1 x_W3_J8 + 1 x_W4_J8 = 1
  assign_J9: + 1 x_W1_J9 + 1 x_W2_J9 + 1 x_W3_J9 + 1 x_W4_J9 = 1
  capacity_W1: + 4 x_W1_J1 + 5 x_W1_J2 + 7 x_W1_J3 + 8 x_W1_J4 + 8 x_W1_J5 + 7 x_W1_J6 + 8 x_W1_J7 + 9 x_W1_J8 + 8 x_W1_J9 <= 15
  capacity_W2: + 7 x_W2_J1 + 8 x_W2_J2 + 4 x_W2_J3 + 5 x_W2_J4 + 8 x_W2_J5 + 8 x_W2_J6 + 7 x_W2_J7 + 8 x_W2_J8 + 9 x_W2_J9 <= 14
  capacity_W3: + 8 x_W3_J1 + 7 x_W3_J2 + 8 x_W3_J3 + 7 x_W3_J4 + 5 x_W3_J5 + 4 x_W3_J6 + 7 x_W3_J7 + 8 x_W3_J8 + 7 x_W3_J9 <= 16
  capacity_W4: + 8 x_W4_J1 + 8 x_W4_J2 + 9 x_W4_J3 + 8 x_W4_J4 + 7 x_W4_J5 + 7 x_W4_J6 + 4 x_W4_J7 + 5 x_W4_J8 + 4 x_W4_J9 <= 13
Bounds
  0 <= x_W1_J1 <= 1
  0 <= x_W1_J2 <= 1
  0 <= x_W1_J3 <= 1
  0 <= x_W1_J4 <= 1
  0 <= x_W1_J5 <= 1
  0 <= x_W1_J6 <= 1
  0 <= x_W1_J7 <= 1
  0 <= x_W1_J8 <= 1
  0 <= x_W1_J9 <= 1
  0 <= x_W2_J1 <= 1
  0 <= x_W2_J2 <= 1
  0 <= x_W2_J3 <= 1
  0 <= x_W2_J4 <= 1
  0 <= x_W2_J5 <= 1
  0 <= x_W2_J6 <= 1
  0 <= x_W2_J7 <= 1
  0 <= x_W2_J8 <= 1
  0 <= x_W2_J9 <= 1
  0 <= x_W3_J1 <= 1
  0 <= x_W3_J2 <= 1
  0 <= x_W3_J3 <= 1
  0 <= x_W3_J4 <= 1
  0 <= x_W3_J5 <= 1
  0 <= x_W3_J6 <= 1
  0 <= x_W3_J7 <= 1
  0 <= x_W3_J8 <= 1
  0 <= x_W3_J9 <= 1
  0 <= x_W4_J1 <= 1
  0 <= x_W4_J2 <= 1
  0 <= x_W4_J3 <= 1
  0 <= x_W4_J4 <= 1
  0 <= x_W4_J5 <= 1
  0 <= x_W4_J6 <= 1
  0 <= x_W4_J7 <= 1
  0 <= x_W4_J8 <= 1
  0 <= x_W4_J9 <= 1
Binaries
  x_W1_J1 x_W1_J2 x_W1_J3 x_W1_J4 x_W1_J5 x_W1_J6 x_W1_J7 x_W1_J8 x_W1_J9 x_W2_J1 x_W2_J2 x_W2_J3 x_W2_J4 x_W2_J5 x_W2_J6 x_W2_J7 x_W2_J8 x_W2_J9 x_W3_J1 x_W3_J2 x_W3_J3 x_W3_J4 x_W3_J5 x_W3_J6 x_W3_J7 x_W3_J8 x_W3_J9 x_W4_J1 x_W4_J2 x_W4_J3 x_W4_J4 x_W4_J5 x_W4_J6 x_W4_J7 x_W4_J8 x_W4_J9
End
