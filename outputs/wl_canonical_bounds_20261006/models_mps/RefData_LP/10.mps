* Signature: 0x99a0afbc10304051
NAME _copy
ROWS
 N  OBJ
 E  Total_Demand
 L  Linking[A]
 L  Linking[B]
 L  Linking[C]
 L  Linking[D]
 G  __wl_bound_00001_lower
 G  __wl_bound_00002_lower
 G  __wl_bound_00003_lower
 G  __wl_bound_00004_lower
 G  __wl_bound_00005_lower
 L  __wl_bound_00005_upper
 G  __wl_bound_00006_lower
 L  __wl_bound_00006_upper
 G  __wl_bound_00007_lower
 L  __wl_bound_00007_upper
 G  __wl_bound_00008_lower
 L  __wl_bound_00008_upper
COLUMNS
    x[A]      OBJ       20
    x[A]      Total_Demand  1
    x[A]      Linking[A]  1
    x[A]      __wl_bound_00001_lower  1
    x[B]      OBJ       24
    x[B]      Total_Demand  1
    x[B]      Linking[B]  1
    x[B]      __wl_bound_00002_lower  1
    x[C]      OBJ       16
    x[C]      Total_Demand  1
    x[C]      Linking[C]  1
    x[C]      __wl_bound_00003_lower  1
    x[D]      OBJ       28
    x[D]      Total_Demand  1
    x[D]      Linking[D]  1
    x[D]      __wl_bound_00004_lower  1
    MARKER    'MARKER'                 'INTORG'
    y[A]      OBJ       1000
    y[A]      Linking[A]  -900
    y[A]      __wl_bound_00005_lower  1
    y[A]      __wl_bound_00005_upper  1
    y[B]      OBJ       920
    y[B]      Linking[B]  -1000
    y[B]      __wl_bound_00006_lower  1
    y[B]      __wl_bound_00006_upper  1
    y[C]      OBJ       800
    y[C]      Linking[C]  -1200
    y[C]      __wl_bound_00007_lower  1
    y[C]      __wl_bound_00007_upper  1
    y[D]      OBJ       700
    y[D]      Linking[D]  -1600
    y[D]      __wl_bound_00008_lower  1
    y[D]      __wl_bound_00008_upper  1
    MARKER    'MARKER'                 'INTEND'
RHS
    RHS1      Total_Demand  2000
    RHS1      __wl_bound_00005_upper  1
    RHS1      __wl_bound_00006_upper  1
    RHS1      __wl_bound_00007_upper  1
    RHS1      __wl_bound_00008_upper  1
BOUNDS
 FR BND1      x[A]    
 FR BND1      x[B]    
 FR BND1      x[C]    
 FR BND1      x[D]    
 BV BND1      y[A]    
 BV BND1      y[B]    
 BV BND1      y[C]    
 BV BND1      y[D]    
ENDATA
