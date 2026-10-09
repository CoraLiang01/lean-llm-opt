\ Canonical native-bound formulation; bounds excluded from WL graph.
Minimize
  obj: + 120 z_A + 150 z_B + 180 z_C + 160 z_D + 210 z_E + 140 z_F + 170 z_G + 200 z_H + 260 z_I
Subject To
  precedence_A_C: + 1 z_A + 1 s_C - 1 s_A >= 4
  precedence_A_D: + 1 z_A - 1 s_A + 1 s_D >= 4
  precedence_B_E: + 1 z_B + 1 s_E - 1 s_B >= 7
  precedence_C_F: + 1 z_C - 1 s_C + 1 s_F >= 6
  precedence_D_F: + 1 z_D - 1 s_D + 1 s_F >= 5
  precedence_D_G: + 1 z_D - 1 s_D + 1 s_G >= 5
  precedence_E_G: + 1 z_E - 1 s_E + 1 s_G >= 4
  precedence_F_H: + 1 z_F - 1 s_F + 1 s_H >= 7
  precedence_G_H: + 1 z_G - 1 s_G + 1 s_H >= 6
  precedence_H_I: + 1 z_H - 1 s_H + 1 s_I >= 5
  completion_A: + 1 z_A - 1 s_A + 1 T >= 4
  completion_B: + 1 z_B - 1 s_B + 1 T >= 7
  completion_C: + 1 z_C - 1 s_C + 1 T >= 6
  completion_D: + 1 z_D - 1 s_D + 1 T >= 5
  completion_E: + 1 z_E - 1 s_E + 1 T >= 4
  completion_F: + 1 z_F - 1 s_F + 1 T >= 7
  completion_G: + 1 z_G - 1 s_G + 1 T >= 6
  completion_H: + 1 z_H - 1 s_H + 1 T >= 5
  completion_I: + 1 z_I - 1 s_I + 1 T >= 3
Bounds
  0 <= z_A <= 1
  0 <= z_B <= 2
  0 <= z_C <= 2
  0 <= z_D <= 2
  0 <= z_E <= 2
  0 <= z_F <= 2
  0 <= z_G <= 2
  0 <= z_H <= 2
  0 <= z_I <= 1
  0 <= s_C <= +inf
  0 <= s_A <= +inf
  0 <= s_D <= +inf
  0 <= s_E <= +inf
  0 <= s_B <= +inf
  0 <= s_F <= +inf
  0 <= s_G <= +inf
  0 <= s_H <= +inf
  0 <= s_I <= +inf
  0 <= T <= 22
Generals
  z_A z_B z_C z_D z_E z_F z_G z_H z_I
End
