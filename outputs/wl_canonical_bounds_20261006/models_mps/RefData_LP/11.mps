* Signature: 0x500241514cedd73a
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
 G  __wl_bound_00001_lower
 G  __wl_bound_00002_lower
 G  __wl_bound_00003_lower
 G  __wl_bound_00004_lower
 G  __wl_bound_00005_lower
 G  __wl_bound_00006_lower
 G  __wl_bound_00007_lower
 G  __wl_bound_00008_lower
 G  __wl_bound_00009_lower
 G  __wl_bound_00010_lower
 G  __wl_bound_00011_lower
 G  __wl_bound_00012_lower
 G  __wl_bound_00013_lower
COLUMNS
    N         OBJ       1
    N         Requirement_1  -1
    N         Requirement_2  -1
    N         Requirement_3  -1
    N         __wl_bound_00001_lower  1
    x[A,1]    Hours_A   1
    x[A,1]    Requirement_1  10
    x[A,1]    __wl_bound_00002_lower  1
    x[A,2]    Hours_A   1
    x[A,2]    Requirement_2  15
    x[A,2]    __wl_bound_00003_lower  1
    x[A,3]    Hours_A   1
    x[A,3]    Requirement_3  5
    x[A,3]    __wl_bound_00004_lower  1
    x[B,1]    Hours_B   1
    x[B,1]    Requirement_1  15
    x[B,1]    __wl_bound_00005_lower  1
    x[B,2]    Hours_B   1
    x[B,2]    Requirement_2  10
    x[B,2]    __wl_bound_00006_lower  1
    x[B,3]    Hours_B   1
    x[B,3]    Requirement_3  5
    x[B,3]    __wl_bound_00007_lower  1
    x[C,1]    Hours_C   1
    x[C,1]    Requirement_1  20
    x[C,1]    __wl_bound_00008_lower  1
    x[C,2]    Hours_C   1
    x[C,2]    Requirement_2  5
    x[C,2]    __wl_bound_00009_lower  1
    x[C,3]    Hours_C   1
    x[C,3]    Requirement_3  10
    x[C,3]    __wl_bound_00010_lower  1
    x[D,1]    Hours_D   1
    x[D,1]    Requirement_1  10
    x[D,1]    __wl_bound_00011_lower  1
    x[D,2]    Hours_D   1
    x[D,2]    Requirement_2  15
    x[D,2]    __wl_bound_00012_lower  1
    x[D,3]    Hours_D   1
    x[D,3]    Requirement_3  20
    x[D,3]    __wl_bound_00013_lower  1
RHS
    RHS1      Hours_A   100
    RHS1      Hours_B   150
    RHS1      Hours_C   80
    RHS1      Hours_D   200
BOUNDS
 FR BND1      N       
 FR BND1      x[A,1]  
 FR BND1      x[A,2]  
 FR BND1      x[A,3]  
 FR BND1      x[B,1]  
 FR BND1      x[B,2]  
 FR BND1      x[B,3]  
 FR BND1      x[C,1]  
 FR BND1      x[C,2]  
 FR BND1      x[C,3]  
 FR BND1      x[D,1]  
 FR BND1      x[D,2]  
 FR BND1      x[D,3]  
ENDATA
