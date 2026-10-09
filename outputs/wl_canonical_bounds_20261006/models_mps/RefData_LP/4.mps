* Signature: 0x3dad79d207a7548
NAME _copy
OBJSENSE MAX
ROWS
 N  OBJ
 G  min_batch[1]
 G  min_batch[2]
 G  min_batch[3]
 L  linking[1]
 L  linking[2]
 L  linking[3]
 L  shared_production_days
 G  __wl_bound_00001_lower
 L  __wl_bound_00001_upper
 G  __wl_bound_00002_lower
 L  __wl_bound_00002_upper
 G  __wl_bound_00003_lower
 L  __wl_bound_00003_upper
 G  __wl_bound_00004_lower
 L  __wl_bound_00004_upper
 G  __wl_bound_00005_lower
 L  __wl_bound_00005_upper
 G  __wl_bound_00006_lower
 L  __wl_bound_00006_upper
COLUMNS
    MARKER    'MARKER'                 'INTORG'
    x[1]      OBJ       39.62
    x[1]      min_batch[1]  1
    x[1]      linking[1]  1
    x[1]      shared_production_days  0.00170648464163823
    x[1]      __wl_bound_00001_lower  1
    x[1]      __wl_bound_00001_upper  1
    x[2]      OBJ       35.98
    x[2]      min_batch[2]  1
    x[2]      linking[2]  1
    x[2]      shared_production_days  0.00303951367781155
    x[2]      __wl_bound_00002_lower  1
    x[2]      __wl_bound_00002_upper  1
    x[3]      OBJ       37.96
    x[3]      min_batch[3]  1
    x[3]      linking[3]  1
    x[3]      shared_production_days  0.00184842883548983
    x[3]      __wl_bound_00003_lower  1
    x[3]      __wl_bound_00003_upper  1
    y[1]      OBJ       -178539
    y[1]      min_batch[1]  -18
    y[1]      linking[1]  -5732
    y[1]      __wl_bound_00004_lower  1
    y[1]      __wl_bound_00004_upper  1
    y[2]      OBJ       -157708
    y[2]      min_batch[2]  -25
    y[2]      linking[2]  -5607
    y[2]      __wl_bound_00005_lower  1
    y[2]      __wl_bound_00005_upper  1
    y[3]      OBJ       -85192
    y[3]      min_batch[3]  -23
    y[3]      linking[3]  -4653
    y[3]      __wl_bound_00006_lower  1
    y[3]      __wl_bound_00006_upper  1
    MARKER    'MARKER'                 'INTEND'
RHS
    RHS1      shared_production_days  22
    RHS1      __wl_bound_00001_upper  5732
    RHS1      __wl_bound_00002_upper  5607
    RHS1      __wl_bound_00003_upper  4653
    RHS1      __wl_bound_00004_upper  1
    RHS1      __wl_bound_00005_upper  1
    RHS1      __wl_bound_00006_upper  1
BOUNDS
 FR BND1      x[1]    
 FR BND1      x[2]    
 FR BND1      x[3]    
 BV BND1      y[1]    
 BV BND1      y[2]    
 BV BND1      y[3]    
ENDATA
