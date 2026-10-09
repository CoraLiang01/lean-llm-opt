* Signature: 0xf6d4cf55367fcdde
NAME obj_copy
ROWS
 N  OBJ
 L  source_supply_S1
 L  source_supply_S2
 L  source_supply_S3
 G  customer_demand_C1
 G  customer_demand_C2
 G  customer_demand_C3
 G  customer_demand_C4
 E  hub_balance_H1
 E  hub_balance_H2
 L  hub_throughput_H1
 L  hub_throughput_H2
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
 G  __wl_bound_00011_lower
 G  __wl_bound_00012_lower
 G  __wl_bound_00013_lower
 G  __wl_bound_00014_lower
COLUMNS
    f_S1_H1   OBJ       2
    f_S1_H1   source_supply_S1  1
    f_S1_H1   hub_balance_H1  1
    f_S1_H1   hub_throughput_H1  1
    f_S1_H1   __wl_bound_00001_lower  1
    f_S1_H2   OBJ       6
    f_S1_H2   source_supply_S1  1
    f_S1_H2   hub_balance_H2  1
    f_S1_H2   hub_throughput_H2  1
    f_S1_H2   __wl_bound_00002_lower  1
    f_S2_H1   OBJ       4
    f_S2_H1   source_supply_S2  1
    f_S2_H1   hub_balance_H1  1
    f_S2_H1   hub_throughput_H1  1
    f_S2_H1   __wl_bound_00003_lower  1
    f_S2_H2   OBJ       3
    f_S2_H2   source_supply_S2  1
    f_S2_H2   hub_balance_H2  1
    f_S2_H2   hub_throughput_H2  1
    f_S2_H2   __wl_bound_00004_lower  1
    f_S3_H1   OBJ       7
    f_S3_H1   source_supply_S3  1
    f_S3_H1   hub_balance_H1  1
    f_S3_H1   hub_throughput_H1  1
    f_S3_H1   __wl_bound_00005_lower  1
    f_S3_H2   OBJ       2
    f_S3_H2   source_supply_S3  1
    f_S3_H2   hub_balance_H2  1
    f_S3_H2   hub_throughput_H2  1
    f_S3_H2   __wl_bound_00006_lower  1
    f_H1_C1   OBJ       3
    f_H1_C1   customer_demand_C1  1
    f_H1_C1   hub_balance_H1  -1
    f_H1_C1   __wl_bound_00007_lower  1
    f_H1_C2   OBJ       4
    f_H1_C2   customer_demand_C2  1
    f_H1_C2   hub_balance_H1  -1
    f_H1_C2   __wl_bound_00008_lower  1
    f_H1_C3   OBJ       7
    f_H1_C3   customer_demand_C3  1
    f_H1_C3   hub_balance_H1  -1
    f_H1_C3   __wl_bound_00009_lower  1
    f_H1_C4   OBJ       8
    f_H1_C4   customer_demand_C4  1
    f_H1_C4   hub_balance_H1  -1
    f_H1_C4   __wl_bound_00010_lower  1
    f_H2_C1   OBJ       8
    f_H2_C1   customer_demand_C1  1
    f_H2_C1   hub_balance_H2  -1
    f_H2_C1   __wl_bound_00011_lower  1
    f_H2_C2   OBJ       6
    f_H2_C2   customer_demand_C2  1
    f_H2_C2   hub_balance_H2  -1
    f_H2_C2   __wl_bound_00012_lower  1
    f_H2_C3   OBJ       3
    f_H2_C3   customer_demand_C3  1
    f_H2_C3   hub_balance_H2  -1
    f_H2_C3   __wl_bound_00013_lower  1
    f_H2_C4   OBJ       4
    f_H2_C4   customer_demand_C4  1
    f_H2_C4   hub_balance_H2  -1
    f_H2_C4   __wl_bound_00014_lower  1
RHS
    RHS1      source_supply_S1  120
    RHS1      source_supply_S2  100
    RHS1      source_supply_S3  90
    RHS1      customer_demand_C1  70
    RHS1      customer_demand_C2  80
    RHS1      customer_demand_C3  60
    RHS1      customer_demand_C4  90
    RHS1      hub_throughput_H1  170
    RHS1      hub_throughput_H2  160
BOUNDS
 FR BND1      f_S1_H1 
 FR BND1      f_S1_H2 
 FR BND1      f_S2_H1 
 FR BND1      f_S2_H2 
 FR BND1      f_S3_H1 
 FR BND1      f_S3_H2 
 FR BND1      f_H1_C1 
 FR BND1      f_H1_C2 
 FR BND1      f_H1_C3 
 FR BND1      f_H1_C4 
 FR BND1      f_H2_C1 
 FR BND1      f_H2_C2 
 FR BND1      f_H2_C3 
 FR BND1      f_H2_C4 
ENDATA
