Minimize
    obj: 120 z_A + 150 z_B + 180 z_C + 160 z_D + 210 z_E + 140 z_F + 170 z_G + 200 z_H + 260 z_I
    Subject To
    precedence_A_C: s_C - s_A + z_A >= 4
    precedence_A_D: s_D - s_A + z_A >= 4
    precedence_B_E: s_E - s_B + z_B >= 7
    precedence_C_F: s_F - s_C + z_C >= 6
    precedence_D_F: s_F - s_D + z_D >= 5
    precedence_D_G: s_G - s_D + z_D >= 5
    precedence_E_G: s_G - s_E + z_E >= 4
    precedence_F_H: s_H - s_F + z_F >= 7
    precedence_G_H: s_H - s_G + z_G >= 6
    precedence_H_I: s_I - s_H + z_H >= 5
    completion_A: T - s_A + z_A >= 4
    completion_B: T - s_B + z_B >= 7
    completion_C: T - s_C + z_C >= 6
    completion_D: T - s_D + z_D >= 5
    completion_E: T - s_E + z_E >= 4
    completion_F: T - s_F + z_F >= 7
    completion_G: T - s_G + z_G >= 6
    completion_H: T - s_H + z_H >= 5
    completion_I: T - s_I + z_I >= 3
    project_deadline: T <= 22
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
    T >= 0
    0 <= z_A <= 1
    0 <= z_B <= 2
    0 <= z_C <= 2
    0 <= z_D <= 2
    0 <= z_E <= 2
    0 <= z_F <= 2
    0 <= z_G <= 2
    0 <= z_H <= 2
    0 <= z_I <= 1
    Generals
    z_A z_B z_C z_D z_E z_F z_G z_H z_I
    End
