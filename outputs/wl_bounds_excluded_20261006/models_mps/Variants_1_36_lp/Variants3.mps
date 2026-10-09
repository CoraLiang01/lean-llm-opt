* Signature: 0x2163608cdd81781
NAME obj_copy
ROWS
 N  OBJ
 G  precedence_A_C
 G  precedence_A_D
 G  precedence_B_E
 G  precedence_B_F
 G  precedence_C_G
 G  precedence_D_G
 G  precedence_D_H
 G  precedence_E_H
 G  precedence_F_I
 G  precedence_G_J
 G  precedence_H_J
 G  precedence_H_K
 G  precedence_I_K
 G  precedence_J_L
 G  precedence_K_L
 G  completion_A
 G  completion_B
 G  completion_C
 G  completion_D
 G  completion_E
 G  completion_F
 G  completion_G
 G  completion_H
 G  completion_I
 G  completion_J
 G  completion_K
 G  completion_L
COLUMNS
    MARKER    'MARKER'                 'INTORG'
    z_A       OBJ       300
    z_A       precedence_A_C  1
    z_A       precedence_A_D  1
    z_A       completion_A  1
    z_B       OBJ       250
    z_B       precedence_B_E  1
    z_B       precedence_B_F  1
    z_B       completion_B  1
    z_C       OBJ       180
    z_C       precedence_C_G  1
    z_C       completion_C  1
    z_D       OBJ       220
    z_D       precedence_D_G  1
    z_D       precedence_D_H  1
    z_D       completion_D  1
    z_E       OBJ       160
    z_E       precedence_E_H  1
    z_E       completion_E  1
    z_F       OBJ       140
    z_F       precedence_F_I  1
    z_F       completion_F  1
    z_G       OBJ       210
    z_G       precedence_G_J  1
    z_G       completion_G  1
    z_H       OBJ       190
    z_H       precedence_H_J  1
    z_H       precedence_H_K  1
    z_H       completion_H  1
    z_I       OBJ       170
    z_I       precedence_I_K  1
    z_I       completion_I  1
    z_J       OBJ       260
    z_J       precedence_J_L  1
    z_J       completion_J  1
    z_K       OBJ       150
    z_K       precedence_K_L  1
    z_K       completion_K  1
    z_L       OBJ       320
    z_L       completion_L  1
    MARKER    'MARKER'                 'INTEND'
    s_C       precedence_A_C  1
    s_C       precedence_C_G  -1
    s_C       completion_C  -1
    s_A       precedence_A_C  -1
    s_A       precedence_A_D  -1
    s_A       completion_A  -1
    s_D       precedence_A_D  1
    s_D       precedence_D_G  -1
    s_D       precedence_D_H  -1
    s_D       completion_D  -1
    s_E       precedence_B_E  1
    s_E       precedence_E_H  -1
    s_E       completion_E  -1
    s_B       precedence_B_E  -1
    s_B       precedence_B_F  -1
    s_B       completion_B  -1
    s_F       precedence_B_F  1
    s_F       precedence_F_I  -1
    s_F       completion_F  -1
    s_G       precedence_C_G  1
    s_G       precedence_D_G  1
    s_G       precedence_G_J  -1
    s_G       completion_G  -1
    s_H       precedence_D_H  1
    s_H       precedence_E_H  1
    s_H       precedence_H_J  -1
    s_H       precedence_H_K  -1
    s_H       completion_H  -1
    s_I       precedence_F_I  1
    s_I       precedence_I_K  -1
    s_I       completion_I  -1
    s_J       precedence_G_J  1
    s_J       precedence_H_J  1
    s_J       precedence_J_L  -1
    s_J       completion_J  -1
    s_K       precedence_H_K  1
    s_K       precedence_I_K  1
    s_K       precedence_K_L  -1
    s_K       completion_K  -1
    s_L       precedence_J_L  1
    s_L       precedence_K_L  1
    s_L       completion_L  -1
    T         completion_A  1
    T         completion_B  1
    T         completion_C  1
    T         completion_D  1
    T         completion_E  1
    T         completion_F  1
    T         completion_G  1
    T         completion_H  1
    T         completion_I  1
    T         completion_J  1
    T         completion_K  1
    T         completion_L  1
RHS
    RHS1      precedence_A_C  6
    RHS1      precedence_A_D  6
    RHS1      precedence_B_E  5
    RHS1      precedence_B_F  5
    RHS1      precedence_C_G  7
    RHS1      precedence_D_G  4
    RHS1      precedence_D_H  4
    RHS1      precedence_E_H  6
    RHS1      precedence_F_I  8
    RHS1      precedence_G_J  5
    RHS1      precedence_H_J  7
    RHS1      precedence_H_K  7
    RHS1      precedence_I_K  6
    RHS1      precedence_J_L  4
    RHS1      precedence_K_L  5
    RHS1      completion_A  6
    RHS1      completion_B  5
    RHS1      completion_C  7
    RHS1      completion_D  4
    RHS1      completion_E  6
    RHS1      completion_F  8
    RHS1      completion_G  5
    RHS1      completion_H  7
    RHS1      completion_I  6
    RHS1      completion_J  4
    RHS1      completion_K  5
    RHS1      completion_L  3
BOUNDS
 UP BND1      z_A       2
 UP BND1      z_B       1
 UP BND1      z_C       3
 UP BND1      z_D       1
 UP BND1      z_E       2
 UP BND1      z_F       3
 UP BND1      z_G       2
 UP BND1      z_H       3
 UP BND1      z_I       2
 UP BND1      z_J       1
 UP BND1      z_K       2
 UP BND1      z_L       1
 UP BND1      T         23
ENDATA
