* Signature: 0xaebb8f5525e3d921
NAME _copy
OBJSENSE MAX
ROWS
 N  OBJ
 L  Hours_A 
 L  Hours_B 
 L  Hours_C 
 L  Hours_D 
 G  Requirement_1
 G  Requirement_2
 G  Requirement_3
COLUMNS
    N         OBJ       1
    N         Requirement_1  -1
    N         Requirement_2  -1
    N         Requirement_3  -1
    x[A,1]    Hours_A   1
    x[A,1]    Requirement_1  10
    x[A,2]    Hours_A   1
    x[A,2]    Requirement_2  15
    x[A,3]    Hours_A   1
    x[A,3]    Requirement_3  5
    x[B,1]    Hours_B   1
    x[B,1]    Requirement_1  15
    x[B,2]    Hours_B   1
    x[B,2]    Requirement_2  10
    x[B,3]    Hours_B   1
    x[B,3]    Requirement_3  5
    x[C,1]    Hours_C   1
    x[C,1]    Requirement_1  20
    x[C,2]    Hours_C   1
    x[C,2]    Requirement_2  5
    x[C,3]    Hours_C   1
    x[C,3]    Requirement_3  10
    x[D,1]    Hours_D   1
    x[D,1]    Requirement_1  10
    x[D,2]    Hours_D   1
    x[D,2]    Requirement_2  15
    x[D,3]    Hours_D   1
    x[D,3]    Requirement_3  20
RHS
    RHS1      Hours_A   100
    RHS1      Hours_B   150
    RHS1      Hours_C   80
    RHS1      Hours_D   200
BOUNDS
ENDATA
