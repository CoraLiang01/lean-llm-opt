Minimize
    obj: 180 z_A + 150 z_B + 130 z_C + 210 z_D + 160 z_E + 190 z_F + 140 z_G + 220 z_H + 170 z_I
      + 260 z_J
    Subject To
    precedence_A_C: s_C - s_A + z_A >= 5
    precedence_A_D: s_D - s_A + z_A >= 5
    precedence_B_E: s_E - s_B + z_B >= 6
    precedence_C_F: s_F - s_C + z_C >= 7
    precedence_D_F: s_F - s_D + z_D >= 4
    precedence_D_G: s_G - s_D + z_D >= 4
    precedence_E_G: s_G - s_E + z_E >= 5
    precedence_F_H: s_H - s_F + z_F >= 6
    precedence_G_I: s_I - s_G + z_G >= 7
    precedence_H_J: s_J - s_H + z_H >= 4
    precedence_I_J: s_J - s_I + z_I >= 5
    completion_A: T - s_A + z_A >= 5
    completion_B: T - s_B + z_B >= 6
    completion_C: T - s_C + z_C >= 7
    completion_D: T - s_D + z_D >= 4
    completion_E: T - s_E + z_E >= 5
    completion_F: T - s_F + z_F >= 6
    completion_G: T - s_G + z_G >= 7
    completion_H: T - s_H + z_H >= 4
    completion_I: T - s_I + z_I >= 5
    completion_J: T - s_J + z_J >= 3
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
    T >= 0
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
    Generals
    z_A z_B z_C z_D z_E z_F z_G z_H z_I z_J
    End
