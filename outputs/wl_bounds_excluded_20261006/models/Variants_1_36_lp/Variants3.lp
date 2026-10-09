\ Canonical native-bound formulation; bounds excluded from WL graph.
Minimize
  obj: + 300 z_A + 250 z_B + 180 z_C + 220 z_D + 160 z_E + 140 z_F + 210 z_G + 190 z_H + 170 z_I + 260 z_J + 150 z_K + 320 z_L
Subject To
  precedence_A_C: + 1 z_A + 1 s_C - 1 s_A >= 6
  precedence_A_D: + 1 z_A - 1 s_A + 1 s_D >= 6
  precedence_B_E: + 1 z_B + 1 s_E - 1 s_B >= 5
  precedence_B_F: + 1 z_B - 1 s_B + 1 s_F >= 5
  precedence_C_G: + 1 z_C - 1 s_C + 1 s_G >= 7
  precedence_D_G: + 1 z_D - 1 s_D + 1 s_G >= 4
  precedence_D_H: + 1 z_D - 1 s_D + 1 s_H >= 4
  precedence_E_H: + 1 z_E - 1 s_E + 1 s_H >= 6
  precedence_F_I: + 1 z_F - 1 s_F + 1 s_I >= 8
  precedence_G_J: + 1 z_G - 1 s_G + 1 s_J >= 5
  precedence_H_J: + 1 z_H - 1 s_H + 1 s_J >= 7
  precedence_H_K: + 1 z_H - 1 s_H + 1 s_K >= 7
  precedence_I_K: + 1 z_I - 1 s_I + 1 s_K >= 6
  precedence_J_L: + 1 z_J - 1 s_J + 1 s_L >= 4
  precedence_K_L: + 1 z_K - 1 s_K + 1 s_L >= 5
  completion_A: + 1 z_A - 1 s_A + 1 T >= 6
  completion_B: + 1 z_B - 1 s_B + 1 T >= 5
  completion_C: + 1 z_C - 1 s_C + 1 T >= 7
  completion_D: + 1 z_D - 1 s_D + 1 T >= 4
  completion_E: + 1 z_E - 1 s_E + 1 T >= 6
  completion_F: + 1 z_F - 1 s_F + 1 T >= 8
  completion_G: + 1 z_G - 1 s_G + 1 T >= 5
  completion_H: + 1 z_H - 1 s_H + 1 T >= 7
  completion_I: + 1 z_I - 1 s_I + 1 T >= 6
  completion_J: + 1 z_J - 1 s_J + 1 T >= 4
  completion_K: + 1 z_K - 1 s_K + 1 T >= 5
  completion_L: + 1 z_L - 1 s_L + 1 T >= 3
Bounds
  0 <= z_A <= 2
  0 <= z_B <= 1
  0 <= z_C <= 3
  0 <= z_D <= 1
  0 <= z_E <= 2
  0 <= z_F <= 3
  0 <= z_G <= 2
  0 <= z_H <= 3
  0 <= z_I <= 2
  0 <= z_J <= 1
  0 <= z_K <= 2
  0 <= z_L <= 1
  0 <= s_C <= +inf
  0 <= s_A <= +inf
  0 <= s_D <= +inf
  0 <= s_E <= +inf
  0 <= s_B <= +inf
  0 <= s_F <= +inf
  0 <= s_G <= +inf
  0 <= s_H <= +inf
  0 <= s_I <= +inf
  0 <= s_J <= +inf
  0 <= s_K <= +inf
  0 <= s_L <= +inf
  0 <= T <= 23
Generals
  z_A z_B z_C z_D z_E z_F z_G z_H z_I z_J z_K z_L
End
