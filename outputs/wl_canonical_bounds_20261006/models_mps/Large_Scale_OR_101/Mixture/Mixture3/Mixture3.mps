* Signature: 0xfbab77231ccd8d8a
NAME obj_copy
OBJSENSE MAX
ROWS
 N  OBJ
 L  capacity_A1
 L  capacity_A2
 L  capacity_B1
 L  capacity_B2
 E  flow_I  
 E  flow_II 
 E  flow_III
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
 L  __wl_bound_00010_upper
COLUMNS
    x_A1_I    OBJ       0.75
    x_A1_I    capacity_A1  5
    x_A1_I    flow_I    1
    x_A1_I    __wl_bound_00001_lower  1
    x_A1_II   OBJ       1.15
    x_A1_II   capacity_A1  10
    x_A1_II   flow_II   1
    x_A1_II   __wl_bound_00002_lower  1
    x_A2_I    OBJ       0.7753
    x_A2_I    capacity_A2  7
    x_A2_I    flow_I    1
    x_A2_I    __wl_bound_00003_lower  1
    x_A2_II   OBJ       1.3611
    x_A2_II   capacity_A2  9
    x_A2_II   flow_II   1
    x_A2_II   __wl_bound_00004_lower  1
    x_A2_III  OBJ       1.9148
    x_A2_III  capacity_A2  12
    x_A2_III  flow_III  1
    x_A2_III  __wl_bound_00005_lower  1
    x_B1_I    OBJ       -0.375
    x_B1_I    capacity_B1  6
    x_B1_I    flow_I    -1
    x_B1_I    __wl_bound_00006_lower  1
    x_B1_II   OBJ       -0.5
    x_B1_II   capacity_B1  8
    x_B1_II   flow_II   -1
    x_B1_II   __wl_bound_00007_lower  1
    x_B2_I    OBJ       -4.4742857142857145e-01
    x_B2_I    capacity_B2  4
    x_B2_I    flow_I    -1
    x_B2_I    __wl_bound_00008_lower  1
    x_B2_III  OBJ       -1.2304285714285714e+00
    x_B2_III  capacity_B2  11
    x_B2_III  flow_III  -1
    x_B2_III  __wl_bound_00009_lower  1
    x_B3_I    OBJ       -0.35
    x_B3_I    flow_I    -1
    x_B3_I    __wl_bound_00010_lower  1
    x_B3_I    __wl_bound_00010_upper  1
RHS
    RHS1      capacity_A1  6000
    RHS1      capacity_A2  10000
    RHS1      capacity_B1  4000
    RHS1      capacity_B2  7000
    RHS1      __wl_bound_00010_upper  5.7142857142857144e+02
BOUNDS
 FR BND1      x_A1_I  
 FR BND1      x_A1_II 
 FR BND1      x_A2_I  
 FR BND1      x_A2_II 
 FR BND1      x_A2_III
 FR BND1      x_B1_I  
 FR BND1      x_B1_II 
 FR BND1      x_B2_I  
 FR BND1      x_B2_III
 FR BND1      x_B3_I  
ENDATA
