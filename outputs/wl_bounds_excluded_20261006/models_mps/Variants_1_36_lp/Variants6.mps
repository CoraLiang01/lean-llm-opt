* Signature: 0xd0a8f9bd3bb58ab3
NAME obj_copy
OBJSENSE MAX
ROWS
 N  OBJ
 E  choose_F1
 E  choose_F2
 E  choose_F3
 E  choose_F4
 E  choose_F5
 E  choose_F6
 L  weight_limit
 L  budget_limit
COLUMNS
    MARKER    'MARKER'                 'INTORG'
    x_F1_O1   OBJ       28
    x_F1_O1   choose_F1  1
    x_F1_O1   weight_limit  8
    x_F1_O1   budget_limit  10
    x_F1_O2   OBJ       34
    x_F1_O2   choose_F1  1
    x_F1_O2   weight_limit  11
    x_F1_O2   budget_limit  13
    x_F1_O3   OBJ       30
    x_F1_O3   choose_F1  1
    x_F1_O3   weight_limit  9
    x_F1_O3   budget_limit  12
    x_F1_O4   OBJ       24
    x_F1_O4   choose_F1  1
    x_F1_O4   weight_limit  7
    x_F1_O4   budget_limit  9
    x_F2_O1   OBJ       25
    x_F2_O1   choose_F2  1
    x_F2_O1   weight_limit  7
    x_F2_O1   budget_limit  9
    x_F2_O2   OBJ       31
    x_F2_O2   choose_F2  1
    x_F2_O2   weight_limit  10
    x_F2_O2   budget_limit  12
    x_F2_O3   OBJ       36
    x_F2_O3   choose_F2  1
    x_F2_O3   weight_limit  12
    x_F2_O3   budget_limit  14
    x_F2_O4   OBJ       29
    x_F2_O4   choose_F2  1
    x_F2_O4   weight_limit  9
    x_F2_O4   budget_limit  11
    x_F3_O1   OBJ       33
    x_F3_O1   choose_F3  1
    x_F3_O1   weight_limit  10
    x_F3_O1   budget_limit  12
    x_F3_O2   OBJ       27
    x_F3_O2   choose_F3  1
    x_F3_O2   weight_limit  8
    x_F3_O2   budget_limit  10
    x_F3_O3   OBJ       38
    x_F3_O3   choose_F3  1
    x_F3_O3   weight_limit  13
    x_F3_O3   budget_limit  15
    x_F3_O4   OBJ       30
    x_F3_O4   choose_F3  1
    x_F3_O4   weight_limit  9
    x_F3_O4   budget_limit  11
    x_F4_O1   OBJ       26
    x_F4_O1   choose_F4  1
    x_F4_O1   weight_limit  6
    x_F4_O1   budget_limit  8
    x_F4_O2   OBJ       35
    x_F4_O2   choose_F4  1
    x_F4_O2   weight_limit  11
    x_F4_O2   budget_limit  13
    x_F4_O3   OBJ       32
    x_F4_O3   choose_F4  1
    x_F4_O3   weight_limit  10
    x_F4_O3   budget_limit  12
    x_F4_O4   OBJ       28
    x_F4_O4   choose_F4  1
    x_F4_O4   weight_limit  8
    x_F4_O4   budget_limit  10
    x_F5_O1   OBJ       30
    x_F5_O1   choose_F5  1
    x_F5_O1   weight_limit  9
    x_F5_O1   budget_limit  11
    x_F5_O2   OBJ       37
    x_F5_O2   choose_F5  1
    x_F5_O2   weight_limit  12
    x_F5_O2   budget_limit  14
    x_F5_O3   OBJ       29
    x_F5_O3   choose_F5  1
    x_F5_O3   weight_limit  8
    x_F5_O3   budget_limit  10
    x_F5_O4   OBJ       34
    x_F5_O4   choose_F5  1
    x_F5_O4   weight_limit  11
    x_F5_O4   budget_limit  13
    x_F6_O1   OBJ       24
    x_F6_O1   choose_F6  1
    x_F6_O1   weight_limit  7
    x_F6_O1   budget_limit  9
    x_F6_O2   OBJ       32
    x_F6_O2   choose_F6  1
    x_F6_O2   weight_limit  10
    x_F6_O2   budget_limit  12
    x_F6_O3   OBJ       36
    x_F6_O3   choose_F6  1
    x_F6_O3   weight_limit  12
    x_F6_O3   budget_limit  14
    x_F6_O4   OBJ       31
    x_F6_O4   choose_F6  1
    x_F6_O4   weight_limit  9
    x_F6_O4   budget_limit  11
    MARKER    'MARKER'                 'INTEND'
RHS
    RHS1      choose_F1  1
    RHS1      choose_F2  1
    RHS1      choose_F3  1
    RHS1      choose_F4  1
    RHS1      choose_F5  1
    RHS1      choose_F6  1
    RHS1      weight_limit  55
    RHS1      budget_limit  70
BOUNDS
 BV BND1      x_F1_O1 
 BV BND1      x_F1_O2 
 BV BND1      x_F1_O3 
 BV BND1      x_F1_O4 
 BV BND1      x_F2_O1 
 BV BND1      x_F2_O2 
 BV BND1      x_F2_O3 
 BV BND1      x_F2_O4 
 BV BND1      x_F3_O1 
 BV BND1      x_F3_O2 
 BV BND1      x_F3_O3 
 BV BND1      x_F3_O4 
 BV BND1      x_F4_O1 
 BV BND1      x_F4_O2 
 BV BND1      x_F4_O3 
 BV BND1      x_F4_O4 
 BV BND1      x_F5_O1 
 BV BND1      x_F5_O2 
 BV BND1      x_F5_O3 
 BV BND1      x_F5_O4 
 BV BND1      x_F6_O1 
 BV BND1      x_F6_O2 
 BV BND1      x_F6_O3 
 BV BND1      x_F6_O4 
ENDATA
