* Signature: 0xb1c9ec146c0dfd4e
NAME _copy
ROWS
 N  OBJ
 E  depart_0
 E  depart_1
 E  depart_2
 E  depart_3
 E  arrive_0
 E  arrive_1
 E  arrive_2
 E  arrive_3
 L  mtz_1_2 
 L  mtz_1_3 
 L  mtz_2_1 
 L  mtz_2_3 
 L  mtz_3_1 
 L  mtz_3_2 
COLUMNS
    MARKER    'MARKER'                 'INTORG'
    X[0,1]    OBJ       28
    X[0,1]    depart_0  1
    X[0,1]    arrive_1  1
    X[0,2]    OBJ       41
    X[0,2]    depart_0  1
    X[0,2]    arrive_2  1
    X[0,3]    OBJ       63
    X[0,3]    depart_0  1
    X[0,3]    arrive_3  1
    X[1,0]    OBJ       28
    X[1,0]    depart_1  1
    X[1,0]    arrive_0  1
    X[1,2]    OBJ       27
    X[1,2]    depart_1  1
    X[1,2]    arrive_2  1
    X[1,2]    mtz_1_2   3
    X[1,3]    OBJ       87
    X[1,3]    depart_1  1
    X[1,3]    arrive_3  1
    X[1,3]    mtz_1_3   3
    X[2,0]    OBJ       41
    X[2,0]    depart_2  1
    X[2,0]    arrive_0  1
    X[2,1]    OBJ       27
    X[2,1]    depart_2  1
    X[2,1]    arrive_1  1
    X[2,1]    mtz_2_1   3
    X[2,3]    OBJ       81
    X[2,3]    depart_2  1
    X[2,3]    arrive_3  1
    X[2,3]    mtz_2_3   3
    X[3,0]    OBJ       63
    X[3,0]    depart_3  1
    X[3,0]    arrive_0  1
    X[3,1]    OBJ       87
    X[3,1]    depart_3  1
    X[3,1]    arrive_1  1
    X[3,1]    mtz_3_1   3
    X[3,2]    OBJ       81
    X[3,2]    depart_3  1
    X[3,2]    arrive_2  1
    X[3,2]    mtz_3_2   3
    MARKER    'MARKER'                 'INTEND'
    U[1]      mtz_1_2   1
    U[1]      mtz_1_3   1
    U[1]      mtz_2_1   -1
    U[1]      mtz_3_1   -1
    U[2]      mtz_1_2   -1
    U[2]      mtz_2_1   1
    U[2]      mtz_2_3   1
    U[2]      mtz_3_2   -1
    U[3]      mtz_1_3   -1
    U[3]      mtz_2_3   -1
    U[3]      mtz_3_1   1
    U[3]      mtz_3_2   1
RHS
    RHS1      depart_0  1
    RHS1      depart_1  1
    RHS1      depart_2  1
    RHS1      depart_3  1
    RHS1      arrive_0  1
    RHS1      arrive_1  1
    RHS1      arrive_2  1
    RHS1      arrive_3  1
    RHS1      mtz_1_2   2
    RHS1      mtz_1_3   2
    RHS1      mtz_2_1   2
    RHS1      mtz_2_3   2
    RHS1      mtz_3_1   2
    RHS1      mtz_3_2   2
BOUNDS
 BV BND1      X[0,1]  
 BV BND1      X[0,2]  
 BV BND1      X[0,3]  
 BV BND1      X[1,0]  
 BV BND1      X[1,2]  
 BV BND1      X[1,3]  
 BV BND1      X[2,0]  
 BV BND1      X[2,1]  
 BV BND1      X[2,3]  
 BV BND1      X[3,0]  
 BV BND1      X[3,1]  
 BV BND1      X[3,2]  
 LO BND1      U[1]      1
 UP BND1      U[1]      3
 LO BND1      U[2]      1
 UP BND1      U[2]      3
 LO BND1      U[3]      1
 UP BND1      U[3]      3
ENDATA
