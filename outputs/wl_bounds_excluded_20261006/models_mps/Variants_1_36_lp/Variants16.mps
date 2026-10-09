* Signature: 0xe89cf2b6497fd9b5
NAME obj_copy
OBJSENSE MAX
ROWS
 N  OBJ
 E  choose_G1
 E  choose_G2
 E  choose_G3
 E  choose_G4
 E  choose_G5
 L  weight_limit
 L  budget_limit
COLUMNS
    MARKER    'MARKER'                 'INTORG'
    x_G1_O1   OBJ       22
    x_G1_O1   choose_G1  1
    x_G1_O1   weight_limit  6
    x_G1_O1   budget_limit  8
    x_G1_O2   OBJ       29
    x_G1_O2   choose_G1  1
    x_G1_O2   weight_limit  9
    x_G1_O2   budget_limit  11
    x_G1_O3   OBJ       31
    x_G1_O3   choose_G1  1
    x_G1_O3   weight_limit  10
    x_G1_O3   budget_limit  13
    x_G1_O4   OBJ       25
    x_G1_O4   choose_G1  1
    x_G1_O4   weight_limit  7
    x_G1_O4   budget_limit  9
    x_G2_O1   OBJ       24
    x_G2_O1   choose_G2  1
    x_G2_O1   weight_limit  7
    x_G2_O1   budget_limit  8
    x_G2_O2   OBJ       33
    x_G2_O2   choose_G2  1
    x_G2_O2   weight_limit  11
    x_G2_O2   budget_limit  13
    x_G2_O3   OBJ       28
    x_G2_O3   choose_G2  1
    x_G2_O3   weight_limit  8
    x_G2_O3   budget_limit  10
    x_G2_O4   OBJ       35
    x_G2_O4   choose_G2  1
    x_G2_O4   weight_limit  12
    x_G2_O4   budget_limit  14
    x_G3_O1   OBJ       30
    x_G3_O1   choose_G3  1
    x_G3_O1   weight_limit  9
    x_G3_O1   budget_limit  12
    x_G3_O2   OBJ       26
    x_G3_O2   choose_G3  1
    x_G3_O2   weight_limit  7
    x_G3_O2   budget_limit  9
    x_G3_O3   OBJ       38
    x_G3_O3   choose_G3  1
    x_G3_O3   weight_limit  13
    x_G3_O3   budget_limit  16
    x_G3_O4   OBJ       34
    x_G3_O4   choose_G3  1
    x_G3_O4   weight_limit  11
    x_G3_O4   budget_limit  13
    x_G4_O1   OBJ       21
    x_G4_O1   choose_G4  1
    x_G4_O1   weight_limit  5
    x_G4_O1   budget_limit  7
    x_G4_O2   OBJ       32
    x_G4_O2   choose_G4  1
    x_G4_O2   weight_limit  10
    x_G4_O2   budget_limit  12
    x_G4_O3   OBJ       36
    x_G4_O3   choose_G4  1
    x_G4_O3   weight_limit  12
    x_G4_O3   budget_limit  15
    x_G4_O4   OBJ       27
    x_G4_O4   choose_G4  1
    x_G4_O4   weight_limit  8
    x_G4_O4   budget_limit  10
    x_G5_O1   OBJ       29
    x_G5_O1   choose_G5  1
    x_G5_O1   weight_limit  8
    x_G5_O1   budget_limit  11
    x_G5_O2   OBJ       37
    x_G5_O2   choose_G5  1
    x_G5_O2   weight_limit  12
    x_G5_O2   budget_limit  15
    x_G5_O3   OBJ       33
    x_G5_O3   choose_G5  1
    x_G5_O3   weight_limit  10
    x_G5_O3   budget_limit  12
    x_G5_O4   OBJ       24
    x_G5_O4   choose_G5  1
    x_G5_O4   weight_limit  6
    x_G5_O4   budget_limit  8
    MARKER    'MARKER'                 'INTEND'
RHS
    RHS1      choose_G1  1
    RHS1      choose_G2  1
    RHS1      choose_G3  1
    RHS1      choose_G4  1
    RHS1      choose_G5  1
    RHS1      weight_limit  48
    RHS1      budget_limit  60
BOUNDS
 BV BND1      x_G1_O1 
 BV BND1      x_G1_O2 
 BV BND1      x_G1_O3 
 BV BND1      x_G1_O4 
 BV BND1      x_G2_O1 
 BV BND1      x_G2_O2 
 BV BND1      x_G2_O3 
 BV BND1      x_G2_O4 
 BV BND1      x_G3_O1 
 BV BND1      x_G3_O2 
 BV BND1      x_G3_O3 
 BV BND1      x_G3_O4 
 BV BND1      x_G4_O1 
 BV BND1      x_G4_O2 
 BV BND1      x_G4_O3 
 BV BND1      x_G4_O4 
 BV BND1      x_G5_O1 
 BV BND1      x_G5_O2 
 BV BND1      x_G5_O3 
 BV BND1      x_G5_O4 
ENDATA
