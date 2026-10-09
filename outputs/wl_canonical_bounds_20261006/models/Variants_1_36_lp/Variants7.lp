\ Canonical explicit-bound formulation for WL comparison.
\ Finite bounds and singleton restrictions merged; original variable domains retained.
Minimize
  obj: + 2 f_S1_H1 + 6 f_S1_H2 + 4 f_S2_H1 + 3 f_S2_H2 + 7 f_S3_H1 + 2 f_S3_H2 + 3 f_H1_C1 + 4 f_H1_C2 + 7 f_H1_C3 + 8 f_H1_C4 + 8 f_H2_C1 + 6 f_H2_C2 + 3 f_H2_C3 + 4 f_H2_C4
Subject To
  source_supply_S1: + 1 f_S1_H1 + 1 f_S1_H2 <= 120
  source_supply_S2: + 1 f_S2_H1 + 1 f_S2_H2 <= 100
  source_supply_S3: + 1 f_S3_H1 + 1 f_S3_H2 <= 90
  customer_demand_C1: + 1 f_H1_C1 + 1 f_H2_C1 >= 70
  customer_demand_C2: + 1 f_H1_C2 + 1 f_H2_C2 >= 80
  customer_demand_C3: + 1 f_H1_C3 + 1 f_H2_C3 >= 60
  customer_demand_C4: + 1 f_H1_C4 + 1 f_H2_C4 >= 90
  hub_balance_H1: + 1 f_S1_H1 + 1 f_S2_H1 + 1 f_S3_H1 - 1 f_H1_C1 - 1 f_H1_C2 - 1 f_H1_C3 - 1 f_H1_C4 = 0
  hub_balance_H2: + 1 f_S1_H2 + 1 f_S2_H2 + 1 f_S3_H2 - 1 f_H2_C1 - 1 f_H2_C2 - 1 f_H2_C3 - 1 f_H2_C4 = 0
  hub_throughput_H1: + 1 f_S1_H1 + 1 f_S2_H1 + 1 f_S3_H1 <= 170
  hub_throughput_H2: + 1 f_S1_H2 + 1 f_S2_H2 + 1 f_S3_H2 <= 160
  __wl_bound_00001_lower: + 1 f_S1_H1 >= 0
  __wl_bound_00002_lower: + 1 f_S1_H2 >= 0
  __wl_bound_00003_lower: + 1 f_S2_H1 >= 0
  __wl_bound_00004_lower: + 1 f_S2_H2 >= 0
  __wl_bound_00005_lower: + 1 f_S3_H1 >= 0
  __wl_bound_00006_lower: + 1 f_S3_H2 >= 0
  __wl_bound_00007_lower: + 1 f_H1_C1 >= 0
  __wl_bound_00008_lower: + 1 f_H1_C2 >= 0
  __wl_bound_00009_lower: + 1 f_H1_C3 >= 0
  __wl_bound_00010_lower: + 1 f_H1_C4 >= 0
  __wl_bound_00011_lower: + 1 f_H2_C1 >= 0
  __wl_bound_00012_lower: + 1 f_H2_C2 >= 0
  __wl_bound_00013_lower: + 1 f_H2_C3 >= 0
  __wl_bound_00014_lower: + 1 f_H2_C4 >= 0
Bounds
  f_S1_H1 free
  f_S1_H2 free
  f_S2_H1 free
  f_S2_H2 free
  f_S3_H1 free
  f_S3_H2 free
  f_H1_C1 free
  f_H1_C2 free
  f_H1_C3 free
  f_H1_C4 free
  f_H2_C1 free
  f_H2_C2 free
  f_H2_C3 free
  f_H2_C4 free
End
