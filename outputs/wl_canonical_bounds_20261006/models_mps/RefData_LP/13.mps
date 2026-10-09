* Signature: 0xedb558ce07cd753b
NAME _copy
ROWS
 N  OBJ0       2  1  0  0
 N  OBJ1       1  1  0  0
 E  Flour_Demand
 E  Rice_Demand
 L  Link_3  
 L  Link_4  
 G  __wl_bound_00001_lower
 G  __wl_bound_00002_lower
 G  __wl_bound_00003_lower
 G  __wl_bound_00004_lower
 G  __wl_bound_00005_lower
 L  __wl_bound_00005_upper
 G  __wl_bound_00006_lower
 L  __wl_bound_00006_upper
COLUMNS
    x[1]      OBJ0      200
    x[1]      Flour_Demand  30
    x[1]      Rice_Demand  20
    x[1]      __wl_bound_00001_lower  1
    x[2]      OBJ0      250
    x[2]      Flour_Demand  20
    x[2]      Rice_Demand  30
    x[2]      __wl_bound_00002_lower  1
    x[3]      OBJ0      380
    x[3]      Flour_Demand  40
    x[3]      Rice_Demand  10
    x[3]      Link_3    1
    x[3]      __wl_bound_00003_lower  1
    x[4]      OBJ0      350
    x[4]      Flour_Demand  10
    x[4]      Rice_Demand  40
    x[4]      Link_4    1
    x[4]      __wl_bound_00004_lower  1
    MARKER    'MARKER'                 'INTORG'
    y[3]      OBJ0      50
    y[3]      OBJ1      1
    y[3]      Link_3    -100
    y[3]      __wl_bound_00005_lower  1
    y[3]      __wl_bound_00005_upper  1
    y[4]      OBJ0      50
    y[4]      OBJ1      1
    y[4]      Link_4    -100
    y[4]      __wl_bound_00006_lower  1
    y[4]      __wl_bound_00006_upper  1
    MARKER    'MARKER'                 'INTEND'
RHS
    RHS1      Flour_Demand  100
    RHS1      Rice_Demand  80
    RHS1      __wl_bound_00005_upper  1
    RHS1      __wl_bound_00006_upper  1
BOUNDS
 FR BND1      x[1]    
 FR BND1      x[2]    
 FR BND1      x[3]    
 FR BND1      x[4]    
 BV BND1      y[3]    
 BV BND1      y[4]    
ENDATA
