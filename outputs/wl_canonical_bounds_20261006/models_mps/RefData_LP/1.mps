* Signature: 0x57a3bbe803fee920
NAME _copy
ROWS
 N  OBJ
 E  worker[MA]
 E  worker[MB]
 E  worker[MC]
 E  project[P1]
 E  project[P2]
 E  project[P3]
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
 G  __wl_bound_00007_lower
 L  __wl_bound_00007_upper
 G  __wl_bound_00008_lower
 L  __wl_bound_00008_upper
 G  __wl_bound_00009_lower
 L  __wl_bound_00009_upper
COLUMNS
    MARKER    'MARKER'                 'INTORG'
    x[MA,P1]  OBJ       3000
    x[MA,P1]  worker[MA]  1
    x[MA,P1]  project[P1]  1
    x[MA,P1]  __wl_bound_00001_lower  1
    x[MA,P1]  __wl_bound_00001_upper  1
    x[MA,P2]  OBJ       3200
    x[MA,P2]  worker[MA]  1
    x[MA,P2]  project[P2]  1
    x[MA,P2]  __wl_bound_00002_lower  1
    x[MA,P2]  __wl_bound_00002_upper  1
    x[MA,P3]  OBJ       3100
    x[MA,P3]  worker[MA]  1
    x[MA,P3]  project[P3]  1
    x[MA,P3]  __wl_bound_00003_lower  1
    x[MA,P3]  __wl_bound_00003_upper  1
    x[MB,P1]  OBJ       2800
    x[MB,P1]  worker[MB]  1
    x[MB,P1]  project[P1]  1
    x[MB,P1]  __wl_bound_00004_lower  1
    x[MB,P1]  __wl_bound_00004_upper  1
    x[MB,P2]  OBJ       3300
    x[MB,P2]  worker[MB]  1
    x[MB,P2]  project[P2]  1
    x[MB,P2]  __wl_bound_00005_lower  1
    x[MB,P2]  __wl_bound_00005_upper  1
    x[MB,P3]  OBJ       2900
    x[MB,P3]  worker[MB]  1
    x[MB,P3]  project[P3]  1
    x[MB,P3]  __wl_bound_00006_lower  1
    x[MB,P3]  __wl_bound_00006_upper  1
    x[MC,P1]  OBJ       2900
    x[MC,P1]  worker[MC]  1
    x[MC,P1]  project[P1]  1
    x[MC,P1]  __wl_bound_00007_lower  1
    x[MC,P1]  __wl_bound_00007_upper  1
    x[MC,P2]  OBJ       3100
    x[MC,P2]  worker[MC]  1
    x[MC,P2]  project[P2]  1
    x[MC,P2]  __wl_bound_00008_lower  1
    x[MC,P2]  __wl_bound_00008_upper  1
    x[MC,P3]  OBJ       3000
    x[MC,P3]  worker[MC]  1
    x[MC,P3]  project[P3]  1
    x[MC,P3]  __wl_bound_00009_lower  1
    x[MC,P3]  __wl_bound_00009_upper  1
    MARKER    'MARKER'                 'INTEND'
RHS
    RHS1      worker[MA]  1
    RHS1      worker[MB]  1
    RHS1      worker[MC]  1
    RHS1      project[P1]  1
    RHS1      project[P2]  1
    RHS1      project[P3]  1
    RHS1      __wl_bound_00001_upper  1
    RHS1      __wl_bound_00002_upper  1
    RHS1      __wl_bound_00003_upper  1
    RHS1      __wl_bound_00004_upper  1
    RHS1      __wl_bound_00005_upper  1
    RHS1      __wl_bound_00006_upper  1
    RHS1      __wl_bound_00007_upper  1
    RHS1      __wl_bound_00008_upper  1
    RHS1      __wl_bound_00009_upper  1
BOUNDS
 BV BND1      x[MA,P1]
 BV BND1      x[MA,P2]
 BV BND1      x[MA,P3]
 BV BND1      x[MB,P1]
 BV BND1      x[MB,P2]
 BV BND1      x[MB,P3]
 BV BND1      x[MC,P1]
 BV BND1      x[MC,P2]
 BV BND1      x[MC,P3]
ENDATA
