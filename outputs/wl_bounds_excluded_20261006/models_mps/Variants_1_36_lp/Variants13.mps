* Signature: 0xe52a8ffb359bb37d
NAME obj_copy
ROWS
 N  OBJ
 G  precedence_A_C
 G  precedence_A_D
 G  precedence_B_E
 G  precedence_C_F
 G  precedence_D_F
 G  precedence_D_G
 G  precedence_E_G
 G  precedence_F_H
 G  precedence_G_I
 G  precedence_H_J
 G  precedence_I_J
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
COLUMNS
    MARKER    'MARKER'                 'INTORG'
    z_A       OBJ       180
    z_A       precedence_A_C  1
    z_A       precedence_A_D  1
    z_A       completion_A  1
    z_B       OBJ       150
    z_B       precedence_B_E  1
    z_B       completion_B  1
    z_C       OBJ       130
    z_C       precedence_C_F  1
    z_C       completion_C  1
    z_D       OBJ       210
    z_D       precedence_D_F  1
    z_D       precedence_D_G  1
    z_D       completion_D  1
    z_E       OBJ       160
    z_E       precedence_E_G  1
    z_E       completion_E  1
    z_F       OBJ       190
    z_F       precedence_F_H  1
    z_F       completion_F  1
    z_G       OBJ       140
    z_G       precedence_G_I  1
    z_G       completion_G  1
    z_H       OBJ       220
    z_H       precedence_H_J  1
    z_H       completion_H  1
    z_I       OBJ       170
    z_I       precedence_I_J  1
    z_I       completion_I  1
    z_J       OBJ       260
    z_J       completion_J  1
    MARKER    'MARKER'                 'INTEND'
    s_C       precedence_A_C  1
    s_C       precedence_C_F  -1
    s_C       completion_C  -1
    s_A       precedence_A_C  -1
    s_A       precedence_A_D  -1
    s_A       completion_A  -1
    s_D       precedence_A_D  1
    s_D       precedence_D_F  -1
    s_D       precedence_D_G  -1
    s_D       completion_D  -1
    s_E       precedence_B_E  1
    s_E       precedence_E_G  -1
    s_E       completion_E  -1
    s_B       precedence_B_E  -1
    s_B       completion_B  -1
    s_F       precedence_C_F  1
    s_F       precedence_D_F  1
    s_F       precedence_F_H  -1
    s_F       completion_F  -1
    s_G       precedence_D_G  1
    s_G       precedence_E_G  1
    s_G       precedence_G_I  -1
    s_G       completion_G  -1
    s_H       precedence_F_H  1
    s_H       precedence_H_J  -1
    s_H       completion_H  -1
    s_I       precedence_G_I  1
    s_I       precedence_I_J  -1
    s_I       completion_I  -1
    s_J       precedence_H_J  1
    s_J       precedence_I_J  1
    s_J       completion_J  -1
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
RHS
    RHS1      precedence_A_C  5
    RHS1      precedence_A_D  5
    RHS1      precedence_B_E  6
    RHS1      precedence_C_F  7
    RHS1      precedence_D_F  4
    RHS1      precedence_D_G  4
    RHS1      precedence_E_G  5
    RHS1      precedence_F_H  6
    RHS1      precedence_G_I  7
    RHS1      precedence_H_J  4
    RHS1      precedence_I_J  5
    RHS1      completion_A  5
    RHS1      completion_B  6
    RHS1      completion_C  7
    RHS1      completion_D  4
    RHS1      completion_E  5
    RHS1      completion_F  6
    RHS1      completion_G  7
    RHS1      completion_H  4
    RHS1      completion_I  5
    RHS1      completion_J  3
BOUNDS
 UP BND1      z_A       2
 UP BND1      z_B       2
 UP BND1      z_C       3
 UP BND1      z_D       1
 UP BND1      z_E       2
 UP BND1      z_F       2
 UP BND1      z_G       2
 UP BND1      z_H       2
 UP BND1      z_I       2
 UP BND1      z_J       1
 UP BND1      T         23
ENDATA
