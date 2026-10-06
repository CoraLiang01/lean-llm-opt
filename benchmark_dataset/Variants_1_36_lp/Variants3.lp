Minimize
    obj: 300 z_A + 250 z_B + 180 z_C + 220 z_D + 160 z_E + 140 z_F + 210 z_G + 190 z_H + 170 z_I
      + 260 z_J + 150 z_K + 320 z_L
    Subject To
    precedence_A_C: s_C - s_A + z_A >= 6
    precedence_A_D: s_D - s_A + z_A >= 6
    precedence_B_E: s_E - s_B + z_B >= 5
    precedence_B_F: s_F - s_B + z_B >= 5
    precedence_C_G: s_G - s_C + z_C >= 7
    precedence_D_G: s_G - s_D + z_D >= 4
    precedence_D_H: s_H - s_D + z_D >= 4
    precedence_E_H: s_H - s_E + z_E >= 6
    precedence_F_I: s_I - s_F + z_F >= 8
    precedence_G_J: s_J - s_G + z_G >= 5
    precedence_H_J: s_J - s_H + z_H >= 7
    precedence_H_K: s_K - s_H + z_H >= 7
    precedence_I_K: s_K - s_I + z_I >= 6
    precedence_J_L: s_L - s_J + z_J >= 4
    precedence_K_L: s_L - s_K + z_K >= 5
    completion_A: T - s_A + z_A >= 6
    completion_B: T - s_B + z_B >= 5
    completion_C: T - s_C + z_C >= 7
    completion_D: T - s_D + z_D >= 4
    completion_E: T - s_E + z_E >= 6
    completion_F: T - s_F + z_F >= 8
    completion_G: T - s_G + z_G >= 5
    completion_H: T - s_H + z_H >= 7
    completion_I: T - s_I + z_I >= 6
    completion_J: T - s_J + z_J >= 4
    completion_K: T - s_K + z_K >= 5
    completion_L: T - s_L + z_L >= 3
    project_deadline: T <= 23
    Bounds
    s_A >= 0
    s_B >= 0
    s_C >= 0
    s_D >= 0
    s_E >= 0
    s_F >= 0
    s_G >= 0
    s_H >= 0
    s_I >= 0
    s_J >= 0
    s_K >= 0
    s_L >= 0
    T >= 0
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
    Generals
    z_A z_B z_C z_D z_E z_F z_G z_H z_I z_J z_K z_L
    End
