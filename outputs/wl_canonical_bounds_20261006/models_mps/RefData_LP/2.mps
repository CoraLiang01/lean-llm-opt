* Signature: 0x3ef419b2cfa0124d
NAME _copy
ROWS
 N  OBJ
 E  demand[C1]
 E  demand[C2]
 L  activation[S1]
 L  activation[S2]
 G  __wl_bound_00001_lower
 G  __wl_bound_00002_lower
 G  __wl_bound_00003_lower
 G  __wl_bound_00004_lower
 G  __wl_bound_00005_lower
 L  __wl_bound_00005_upper
 G  __wl_bound_00006_lower
 L  __wl_bound_00006_upper
COLUMNS
    x[S1,C1]  OBJ       9.3734065047577019e+02
    x[S1,C1]  demand[C1]  1
    x[S1,C1]  activation[S1]  1
    x[S1,C1]  __wl_bound_00001_lower  1
    x[S1,C2]  OBJ       8.6930194110278521e+01
    x[S1,C2]  demand[C2]  1
    x[S1,C2]  activation[S1]  1
    x[S1,C2]  __wl_bound_00002_lower  1
    x[S2,C1]  OBJ       49.3801614788785
    x[S2,C1]  demand[C1]  1
    x[S2,C1]  activation[S2]  1
    x[S2,C1]  __wl_bound_00003_lower  1
    x[S2,C2]  OBJ       1.7260621013627840e+03
    x[S2,C2]  demand[C2]  1
    x[S2,C2]  activation[S2]  1
    x[S2,C2]  __wl_bound_00004_lower  1
    MARKER    'MARKER'                 'INTORG'
    y[S1]     OBJ       1.0518150830480550e+02
    y[S1]     activation[S1]  -12810
    y[S1]     __wl_bound_00005_lower  1
    y[S1]     __wl_bound_00005_upper  1
    y[S2]     OBJ       1.1218423885126420e+02
    y[S2]     activation[S2]  -12810
    y[S2]     __wl_bound_00006_lower  1
    y[S2]     __wl_bound_00006_upper  1
    MARKER    'MARKER'                 'INTEND'
RHS
    RHS1      demand[C1]  7564
    RHS1      demand[C2]  5246
    RHS1      __wl_bound_00005_upper  1
    RHS1      __wl_bound_00006_upper  1
BOUNDS
 FR BND1      x[S1,C1]
 FR BND1      x[S1,C2]
 FR BND1      x[S2,C1]
 FR BND1      x[S2,C2]
 BV BND1      y[S1]   
 BV BND1      y[S2]   
ENDATA
