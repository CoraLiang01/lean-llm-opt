* Signature: 0x14c847781b751896
NAME _copy
ROWS
 N  OBJ
 E  Total_Demand
 L  Linking[A]
 L  Linking[B]
 L  Linking[C]
 L  Linking[D]
COLUMNS
    x[A]      OBJ       20
    x[A]      Total_Demand  1
    x[A]      Linking[A]  1
    x[B]      OBJ       24
    x[B]      Total_Demand  1
    x[B]      Linking[B]  1
    x[C]      OBJ       16
    x[C]      Total_Demand  1
    x[C]      Linking[C]  1
    x[D]      OBJ       28
    x[D]      Total_Demand  1
    x[D]      Linking[D]  1
    MARKER    'MARKER'                 'INTORG'
    y[A]      OBJ       1000
    y[A]      Linking[A]  -900
    y[B]      OBJ       920
    y[B]      Linking[B]  -1000
    y[C]      OBJ       800
    y[C]      Linking[C]  -1200
    y[D]      OBJ       700
    y[D]      Linking[D]  -1600
    MARKER    'MARKER'                 'INTEND'
RHS
    RHS1      Total_Demand  2000
BOUNDS
 BV BND1      y[A]    
 BV BND1      y[B]    
 BV BND1      y[C]    
 BV BND1      y[D]    
ENDATA
