* Signature: 0x5da1416d10932dc9
NAME _copy
OBJSENSE MAX
ROWS
 N  OBJ
 L  supply_I
 L  supply_II
 L  supply_III
 L  Red_I_upper_ratio
 G  Red_II_lower_ratio
 L  Yellow_III_upper_ratio
 G  Yellow_I_lower_ratio
 L  Blue_I_upper_ratio
 G  Blue_II_lower_ratio
 G  min_production_Red
 G  __wl_bound_00001_lower
 G  __wl_bound_00002_lower
 G  __wl_bound_00003_lower
 G  __wl_bound_00004_lower
 G  __wl_bound_00005_lower
 G  __wl_bound_00006_lower
 G  __wl_bound_00007_lower
 G  __wl_bound_00008_lower
 G  __wl_bound_00009_lower
COLUMNS
    x[Red,I]  OBJ       -0.5
    x[Red,I]  supply_I  1
    x[Red,I]  Red_I_upper_ratio  0.9
    x[Red,I]  Red_II_lower_ratio  -0.5
    x[Red,I]  min_production_Red  1
    x[Red,I]  __wl_bound_00001_lower  1
    x[Red,II]  OBJ       1
    x[Red,II]  supply_II  1
    x[Red,II]  Red_I_upper_ratio  -0.1
    x[Red,II]  Red_II_lower_ratio  0.5
    x[Red,II]  min_production_Red  1
    x[Red,II]  __wl_bound_00002_lower  1
    x[Red,III]  OBJ       2.5
    x[Red,III]  supply_III  1
    x[Red,III]  Red_I_upper_ratio  -0.1
    x[Red,III]  Red_II_lower_ratio  -0.5
    x[Red,III]  min_production_Red  1
    x[Red,III]  __wl_bound_00003_lower  1
    x[Yellow,I]  OBJ       -1
    x[Yellow,I]  supply_I  1
    x[Yellow,I]  Yellow_III_upper_ratio  -0.7
    x[Yellow,I]  Yellow_I_lower_ratio  0.8
    x[Yellow,I]  __wl_bound_00004_lower  1
    x[Yellow,II]  OBJ       0.5
    x[Yellow,II]  supply_II  1
    x[Yellow,II]  Yellow_III_upper_ratio  -0.7
    x[Yellow,II]  Yellow_I_lower_ratio  -0.2
    x[Yellow,II]  __wl_bound_00005_lower  1
    x[Yellow,III]  OBJ       2
    x[Yellow,III]  supply_III  1
    x[Yellow,III]  Yellow_III_upper_ratio  0.3
    x[Yellow,III]  Yellow_I_lower_ratio  -0.2
    x[Yellow,III]  __wl_bound_00006_lower  1
    x[Blue,I]  OBJ       -1.2
    x[Blue,I]  supply_I  1
    x[Blue,I]  Blue_I_upper_ratio  0.5
    x[Blue,I]  Blue_II_lower_ratio  -0.1
    x[Blue,I]  __wl_bound_00007_lower  1
    x[Blue,II]  OBJ       0.3
    x[Blue,II]  supply_II  1
    x[Blue,II]  Blue_I_upper_ratio  -0.5
    x[Blue,II]  Blue_II_lower_ratio  0.9
    x[Blue,II]  __wl_bound_00008_lower  1
    x[Blue,III]  OBJ       1.8
    x[Blue,III]  supply_III  1
    x[Blue,III]  Blue_I_upper_ratio  -0.5
    x[Blue,III]  Blue_II_lower_ratio  -0.1
    x[Blue,III]  __wl_bound_00009_lower  1
RHS
    RHS1      supply_I  1500
    RHS1      supply_II  2000
    RHS1      supply_III  1000
    RHS1      min_production_Red  2000
BOUNDS
 FR BND1      x[Red,I]
 FR BND1      x[Red,II]
 FR BND1      x[Red,III]
 FR BND1      x[Yellow,I]
 FR BND1      x[Yellow,II]
 FR BND1      x[Yellow,III]
 FR BND1      x[Blue,I]
 FR BND1      x[Blue,II]
 FR BND1      x[Blue,III]
ENDATA
