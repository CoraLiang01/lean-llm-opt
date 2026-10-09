* Signature: 0xb9fb2f63521766bb
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
COLUMNS
    MARKER    'MARKER'                 'INTORG'
    x[1]      OBJ       39.62
    x[1]      min_batch[1]  1
    x[1]      linking[1]  1
    x[1]      shared_production_days  0.00170648464163823
    x[2]      OBJ       35.98
    x[2]      min_batch[2]  1
    x[2]      linking[2]  1
    x[2]      shared_production_days  0.00303951367781155
    x[3]      OBJ       37.96
    x[3]      min_batch[3]  1
    x[3]      linking[3]  1
    x[3]      shared_production_days  0.00184842883548983
    y[1]      OBJ       -178539
    y[1]      min_batch[1]  -18
    y[1]      linking[1]  -5732
    y[2]      OBJ       -157708
    y[2]      min_batch[2]  -25
    y[2]      linking[2]  -5607
    y[3]      OBJ       -85192
    y[3]      min_batch[3]  -23
    y[3]      linking[3]  -4653
    MARKER    'MARKER'                 'INTEND'
RHS
    RHS1      shared_production_days  22
BOUNDS
 UP BND1      x[1]      5732
 UP BND1      x[2]      5607
 UP BND1      x[3]      4653
 BV BND1      y[1]    
 BV BND1      y[2]    
 BV BND1      y[3]    
ENDATA
