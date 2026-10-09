\ Canonical native-bound formulation; bounds excluded from WL graph.
Minimize
  obj: + 180 z_A + 150 z_B + 130 z_C + 210 z_D + 160 z_E + 190 z_F + 140 z_G + 220 z_H + 170 z_I + 260 z_J
Subject To
  precedence_A_C: + 1 z_A + 1 s_C - 1 s_A >= 5
  precedence_A_D: + 1 z_A - 1 s_A + 1 s_D >= 5
  precedence_B_E: + 1 z_B + 1 s_E - 1 s_B >= 6
  precedence_C_F: + 1 z_C - 1 s_C + 1 s_F >= 7
  precedence_D_F: + 1 z_D - 1 s_D + 1 s_F >= 4
  precedence_D_G: + 1 z_D - 1 s_D + 1 s_G >= 4
  precedence_E_G: + 1 z_E - 1 s_E + 1 s_G >= 5
  precedence_F_H: + 1 z_F - 1 s_F + 1 s_H >= 6
  precedence_G_I: + 1 z_G - 1 s_G + 1 s_I >= 7
  precedence_H_J: + 1 z_H - 1 s_H + 1 s_J >= 4
  precedence_I_J: + 1 z_I - 1 s_I + 1 s_J >= 5
  completion_A: + 1 z_A - 1 s_A + 1 T >= 5
  completion_B: + 1 z_B - 1 s_B + 1 T >= 6
  completion_C: + 1 z_C - 1 s_C + 1 T >= 7
  completion_D: + 1 z_D - 1 s_D + 1 T >= 4
  completion_E: + 1 z_E - 1 s_E + 1 T >= 5
  completion_F: + 1 z_F - 1 s_F + 1 T >= 6
  completion_G: + 1 z_G - 1 s_G + 1 T >= 7
  completion_H: + 1 z_H - 1 s_H + 1 T >= 4
  completion_I: + 1 z_I - 1 s_I + 1 T >= 5
  completion_J: + 1 z_J - 1 s_J + 1 T >= 3
Bounds
  0 <= z_A <= 2
  0 <= z_B <= 2
  0 <= z_C <= 3
  0 <= z_D <= 1
  0 <= z_E <= 2
  0 <= z_F <= 2
  0 <= z_G <= 2
  0 <= z_H <= 2
  0 <= z_I <= 2
  0 <= z_J <= 1
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
  0 <= T <= 23
Generals
  z_A z_B z_C z_D z_E z_F z_G z_H z_I z_J
End
