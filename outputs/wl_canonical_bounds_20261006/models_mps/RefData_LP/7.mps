* Signature: 0x2e2a71ae84f4601d
NAME _copy
ROWS
 N  OBJ
 E  Coal_Balance
 E  Power_Balance
 E  Steel_Balance
 G  __wl_bound_00001_lower
 G  __wl_bound_00002_lower
 G  __wl_bound_00003_lower
 L  __wl_bound_00003_upper
COLUMNS
    OutputValue_Coal  Coal_Balance  1
    OutputValue_Coal  Power_Balance  -0.6
    OutputValue_Coal  Steel_Balance  -0.4
    OutputValue_Coal  __wl_bound_00001_lower  1
    OutputValue_Power  Coal_Balance  -0.4
    OutputValue_Power  Power_Balance  0.9
    OutputValue_Power  Steel_Balance  -0.5
    OutputValue_Power  __wl_bound_00002_lower  1
    OutputValue_Steel  Coal_Balance  -0.6
    OutputValue_Steel  Power_Balance  -0.2
    OutputValue_Steel  Steel_Balance  0.8
    OutputValue_Steel  __wl_bound_00003_lower  1
    OutputValue_Steel  __wl_bound_00003_upper  1
RHS
    RHS1      __wl_bound_00003_lower  10000
    RHS1      __wl_bound_00003_upper  10000
BOUNDS
 FR BND1      OutputValue_Coal
 FR BND1      OutputValue_Power
 FR BND1      OutputValue_Steel
ENDATA
