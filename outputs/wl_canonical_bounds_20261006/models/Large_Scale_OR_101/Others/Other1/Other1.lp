\ Canonical explicit-bound formulation for WL comparison.
\ Finite bounds and singleton restrictions merged; original variable domains retained.
Minimize
  obj: + 1 x[0] + 1 x[1] + 1 x[2] + 1 x[3] + 1 x[4] + 1 x[5] + 1 x[6] + 1 x[7] + 1 x[8] + 1 x[9] + 1 x[10] + 1 x[11] + 1 x[12] + 1 x[13] + 1 x[14] + 1 x[15] + 1 x[16] + 1 x[17] + 1 x[18] + 1 x[19] + 1 x[20] + 1 x[21] + 1 x[22] + 1 x[23]
Subject To
  Demand_Hour_00_Met: + 1 x[0] + 1 x[21] + 1 x[22] + 1 x[23] >= 20
  Demand_Hour_01_Met: + 1 x[0] + 1 x[1] + 1 x[22] + 1 x[23] >= 18
  Demand_Hour_02_Met: + 1 x[0] + 1 x[1] + 1 x[2] + 1 x[23] >= 15
  Demand_Hour_03_Met: + 1 x[0] + 1 x[1] + 1 x[2] + 1 x[3] >= 15
  Demand_Hour_04_Met: + 1 x[1] + 1 x[2] + 1 x[3] + 1 x[4] >= 20
  Demand_Hour_05_Met: + 1 x[2] + 1 x[3] + 1 x[4] + 1 x[5] >= 30
  Demand_Hour_06_Met: + 1 x[3] + 1 x[4] + 1 x[5] + 1 x[6] >= 60
  Demand_Hour_07_Met: + 1 x[4] + 1 x[5] + 1 x[6] + 1 x[7] >= 70
  Demand_Hour_08_Met: + 1 x[5] + 1 x[6] + 1 x[7] + 1 x[8] >= 50
  Demand_Hour_09_Met: + 1 x[6] + 1 x[7] + 1 x[8] + 1 x[9] >= 55
  Demand_Hour_10_Met: + 1 x[7] + 1 x[8] + 1 x[9] + 1 x[10] >= 65
  Demand_Hour_11_Met: + 1 x[8] + 1 x[9] + 1 x[10] + 1 x[11] >= 75
  Demand_Hour_12_Met: + 1 x[9] + 1 x[10] + 1 x[11] + 1 x[12] >= 80
  Demand_Hour_13_Met: + 1 x[10] + 1 x[11] + 1 x[12] + 1 x[13] >= 70
  Demand_Hour_14_Met: + 1 x[11] + 1 x[12] + 1 x[13] + 1 x[14] >= 60
  Demand_Hour_15_Met: + 1 x[12] + 1 x[13] + 1 x[14] + 1 x[15] >= 55
  Demand_Hour_16_Met: + 1 x[13] + 1 x[14] + 1 x[15] + 1 x[16] >= 60
  Demand_Hour_17_Met: + 1 x[14] + 1 x[15] + 1 x[16] + 1 x[17] >= 75
  Demand_Hour_18_Met: + 1 x[15] + 1 x[16] + 1 x[17] + 1 x[18] >= 85
  Demand_Hour_19_Met: + 1 x[16] + 1 x[17] + 1 x[18] + 1 x[19] >= 70
  Demand_Hour_20_Met: + 1 x[17] + 1 x[18] + 1 x[19] + 1 x[20] >= 50
  Demand_Hour_21_Met: + 1 x[18] + 1 x[19] + 1 x[20] + 1 x[21] >= 40
  Demand_Hour_22_Met: + 1 x[19] + 1 x[20] + 1 x[21] + 1 x[22] >= 35
  Demand_Hour_23_Met: + 1 x[20] + 1 x[21] + 1 x[22] + 1 x[23] >= 25
  __wl_bound_00001_lower: + 1 x[0] >= 0
  __wl_bound_00002_lower: + 1 x[1] >= 0
  __wl_bound_00003_lower: + 1 x[2] >= 0
  __wl_bound_00004_lower: + 1 x[3] >= 0
  __wl_bound_00005_lower: + 1 x[4] >= 0
  __wl_bound_00006_lower: + 1 x[5] >= 0
  __wl_bound_00007_lower: + 1 x[6] >= 0
  __wl_bound_00008_lower: + 1 x[7] >= 0
  __wl_bound_00009_lower: + 1 x[8] >= 0
  __wl_bound_00010_lower: + 1 x[9] >= 0
  __wl_bound_00011_lower: + 1 x[10] >= 0
  __wl_bound_00012_lower: + 1 x[11] >= 0
  __wl_bound_00013_lower: + 1 x[12] >= 0
  __wl_bound_00014_lower: + 1 x[13] >= 0
  __wl_bound_00015_lower: + 1 x[14] >= 0
  __wl_bound_00016_lower: + 1 x[15] >= 0
  __wl_bound_00017_lower: + 1 x[16] >= 0
  __wl_bound_00018_lower: + 1 x[17] >= 0
  __wl_bound_00019_lower: + 1 x[18] >= 0
  __wl_bound_00020_lower: + 1 x[19] >= 0
  __wl_bound_00021_lower: + 1 x[20] >= 0
  __wl_bound_00022_lower: + 1 x[21] >= 0
  __wl_bound_00023_lower: + 1 x[22] >= 0
  __wl_bound_00024_lower: + 1 x[23] >= 0
Bounds
  x[0] free
  x[1] free
  x[2] free
  x[3] free
  x[4] free
  x[5] free
  x[6] free
  x[7] free
  x[8] free
  x[9] free
  x[10] free
  x[11] free
  x[12] free
  x[13] free
  x[14] free
  x[15] free
  x[16] free
  x[17] free
  x[18] free
  x[19] free
  x[20] free
  x[21] free
  x[22] free
  x[23] free
Generals
  x[0] x[1] x[2] x[3] x[4] x[5] x[6] x[7] x[8] x[9] x[10] x[11] x[12] x[13] x[14] x[15] x[16] x[17] x[18] x[19] x[20] x[21] x[22] x[23]
End
