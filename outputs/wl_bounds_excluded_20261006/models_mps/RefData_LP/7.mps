* Signature: 0x6ccae1a214bf9ecc
NAME _copy
ROWS
 N  OBJ
 E  Coal_Balance
 E  Power_Balance
 E  Steel_Balance
COLUMNS
    OutputValue_Coal  Coal_Balance  1
    OutputValue_Coal  Power_Balance  -0.6
    OutputValue_Coal  Steel_Balance  -0.4
    OutputValue_Power  Coal_Balance  -0.4
    OutputValue_Power  Power_Balance  0.9
    OutputValue_Power  Steel_Balance  -0.5
    OutputValue_Steel  Coal_Balance  -0.6
    OutputValue_Steel  Power_Balance  -0.2
    OutputValue_Steel  Steel_Balance  0.8
RHS
BOUNDS
 FX BND1      OutputValue_Steel  10000
ENDATA
