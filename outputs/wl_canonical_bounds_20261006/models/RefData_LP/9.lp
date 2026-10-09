\ Canonical explicit-bound formulation for WL comparison.
\ Finite bounds and singleton restrictions merged; original variable domains retained.
Maximize
  obj: + 46 x[1,1] + 38 x[1,2] + 41 x[1,3] + 43 x[1,4] + 40 x[1,5] + 37 x[1,6] + 45 x[1,7] + 44 x[2,1] + 39 x[2,2] + 43 x[2,3] + 41 x[2,4] + 39 x[2,5] + 38 x[2,6] + 41 x[2,7] + 47 x[3,1] + 36 x[3,2] + 40 x[3,3] + 45 x[3,4] + 42 x[3,5] + 34 x[3,6] + 43 x[3,7] + 42 x[4,1] + 40 x[4,2] + 41 x[4,3] + 42 x[4,4] + 43 x[4,5] + 39 x[4,6] + 44 x[4,7] + 48 x[5,1] + 38 x[5,2] + 38 x[5,3] + 44 x[5,4] + 43 x[5,5] + 37 x[5,6] + 46 x[5,7] + 46 x[6,1] + 37 x[6,2] + 42 x[6,3] + 41 x[6,4] + 41 x[6,5] + 35 x[6,6] + 45 x[6,7]
Subject To
  Total_Risk_Limit: + 0.13 x[1,1] + 0.12 x[1,2] + 0.17000000000000001 x[1,3] + 0.10000000000000001 x[1,4] + 0.14000000000000001 x[1,5] + 0.13 x[1,6] + 0.17999999999999999 x[1,7] + 0.11 x[2,1] + 0.12 x[2,2] + 0.16 x[2,3] + 0.13 x[2,4] + 0.13 x[2,5] + 0.11 x[2,6] + 0.17000000000000001 x[2,7] + 0.14999999999999999 x[3,1] + 0.13 x[3,2] + 0.19 x[3,3] + 0.13 x[3,4] + 0.14999999999999999 x[3,5] + 0.14000000000000001 x[3,6] + 0.17999999999999999 x[3,7] + 0.12 x[4,1] + 0.14999999999999999 x[4,2] + 0.14999999999999999 x[4,3] + 0.12 x[4,4] + 0.13 x[4,5] + 0.13 x[4,6] + 0.14999999999999999 x[4,7] + 0.17999999999999999 x[5,1] + 0.13 x[5,2] + 0.20999999999999999 x[5,3] + 0.16 x[5,4] + 0.19 x[5,5] + 0.13 x[5,6] + 0.20000000000000001 x[5,7] + 0.14000000000000001 x[6,1] + 0.14999999999999999 x[6,2] + 0.14000000000000001 x[6,3] + 0.14999999999999999 x[6,4]
    + 0.14999999999999999 x[6,5] + 0.12 x[6,6] + 0.14000000000000001 x[6,7] <= 14
  Min_Total_Lending: + 1 x[1,1] + 1 x[1,2] + 1 x[1,3] + 1 x[1,4] + 1 x[1,5] + 1 x[1,6] + 1 x[1,7] + 1 x[2,1] + 1 x[2,2] + 1 x[2,3] + 1 x[2,4] + 1 x[2,5] + 1 x[2,6] + 1 x[2,7] + 1 x[3,1] + 1 x[3,2] + 1 x[3,3] + 1 x[3,4] + 1 x[3,5] + 1 x[3,6] + 1 x[3,7] + 1 x[4,1] + 1 x[4,2] + 1 x[4,3] + 1 x[4,4] + 1 x[4,5] + 1 x[4,6] + 1 x[4,7] + 1 x[5,1] + 1 x[5,2] + 1 x[5,3] + 1 x[5,4] + 1 x[5,5] + 1 x[5,6] + 1 x[5,7] + 1 x[6,1] + 1 x[6,2] + 1 x[6,3] + 1 x[6,4] + 1 x[6,5] + 1 x[6,6] + 1 x[6,7] >= 30
  __wl_bound_00001_lower: + 1 x[1,1] >= 0
  __wl_bound_00001_upper: + 1 x[1,1] <= 7
  __wl_bound_00002_lower: + 1 x[1,2] >= 0
  __wl_bound_00002_upper: + 1 x[1,2] <= 5
  __wl_bound_00003_lower: + 1 x[1,3] >= 0
  __wl_bound_00003_upper: + 1 x[1,3] <= 4
  __wl_bound_00004_lower: + 1 x[1,4] >= 0
  __wl_bound_00004_upper: + 1 x[1,4] <= 8
  __wl_bound_00005_lower: + 1 x[1,5] >= 0
  __wl_bound_00005_upper: + 1 x[1,5] <= 6
  __wl_bound_00006_lower: + 1 x[1,6] >= 0
  __wl_bound_00006_upper: + 1 x[1,6] <= 5
  __wl_bound_00007_lower: + 1 x[1,7] >= 0
  __wl_bound_00007_upper: + 1 x[1,7] <= 7
  __wl_bound_00008_lower: + 1 x[2,1] >= 0
  __wl_bound_00008_upper: + 1 x[2,1] <= 6
  __wl_bound_00009_lower: + 1 x[2,2] >= 0
  __wl_bound_00009_upper: + 1 x[2,2] <= 7
  __wl_bound_00010_lower: + 1 x[2,3] >= 0
  __wl_bound_00010_upper: + 1 x[2,3] <= 5
  __wl_bound_00011_lower: + 1 x[2,4] >= 0
  __wl_bound_00011_upper: + 1 x[2,4] <= 7
  __wl_bound_00012_lower: + 1 x[2,5] >= 0
  __wl_bound_00012_upper: + 1 x[2,5] <= 6
  __wl_bound_00013_lower: + 1 x[2,6] >= 0
  __wl_bound_00013_upper: + 1 x[2,6] <= 4
  __wl_bound_00014_lower: + 1 x[2,7] >= 0
  __wl_bound_00014_upper: + 1 x[2,7] <= 5
  __wl_bound_00015_lower: + 1 x[3,1] >= 0
  __wl_bound_00015_upper: + 1 x[3,1] <= 5
  __wl_bound_00016_lower: + 1 x[3,2] >= 0
  __wl_bound_00016_upper: + 1 x[3,2] <= 6
  __wl_bound_00017_lower: + 1 x[3,3] >= 0
  __wl_bound_00017_upper: + 1 x[3,3] <= 6
  __wl_bound_00018_lower: + 1 x[3,4] >= 0
  __wl_bound_00018_upper: + 1 x[3,4] <= 6
  __wl_bound_00019_lower: + 1 x[3,5] >= 0
  __wl_bound_00019_upper: + 1 x[3,5] <= 5
  __wl_bound_00020_lower: + 1 x[3,6] >= 0
  __wl_bound_00020_upper: + 1 x[3,6] <= 5
  __wl_bound_00021_lower: + 1 x[3,7] >= 0
  __wl_bound_00021_upper: + 1 x[3,7] <= 4
  __wl_bound_00022_lower: + 1 x[4,1] >= 0
  __wl_bound_00022_upper: + 1 x[4,1] <= 8
  __wl_bound_00023_lower: + 1 x[4,2] >= 0
  __wl_bound_00023_upper: + 1 x[4,2] <= 5
  __wl_bound_00024_lower: + 1 x[4,3] >= 0
  __wl_bound_00024_upper: + 1 x[4,3] <= 7
  __wl_bound_00025_lower: + 1 x[4,4] >= 0
  __wl_bound_00025_upper: + 1 x[4,4] <= 8
  __wl_bound_00026_lower: + 1 x[4,5] >= 0
  __wl_bound_00026_upper: + 1 x[4,5] <= 6
  __wl_bound_00027_lower: + 1 x[4,6] >= 0
  __wl_bound_00027_upper: + 1 x[4,6] <= 6
  __wl_bound_00028_lower: + 1 x[4,7] >= 0
  __wl_bound_00028_upper: + 1 x[4,7] <= 5
  __wl_bound_00029_lower: + 1 x[5,1] >= 0
  __wl_bound_00029_upper: + 1 x[5,1] <= 7
  __wl_bound_00030_lower: + 1 x[5,2] >= 0
  __wl_bound_00030_upper: + 1 x[5,2] <= 6
  __wl_bound_00031_lower: + 1 x[5,3] >= 0
  __wl_bound_00031_upper: + 1 x[5,3] <= 6
  __wl_bound_00032_lower: + 1 x[5,4] >= 0
  __wl_bound_00032_upper: + 1 x[5,4] <= 7
  __wl_bound_00033_lower: + 1 x[5,5] >= 0
  __wl_bound_00033_upper: + 1 x[5,5] <= 7
  __wl_bound_00034_lower: + 1 x[5,6] >= 0
  __wl_bound_00034_upper: + 1 x[5,6] <= 4
  __wl_bound_00035_lower: + 1 x[5,7] >= 0
  __wl_bound_00035_upper: + 1 x[5,7] <= 5
  __wl_bound_00036_lower: + 1 x[6,1] >= 0
  __wl_bound_00036_upper: + 1 x[6,1] <= 6
  __wl_bound_00037_lower: + 1 x[6,2] >= 0
  __wl_bound_00037_upper: + 1 x[6,2] <= 5
  __wl_bound_00038_lower: + 1 x[6,3] >= 0
  __wl_bound_00038_upper: + 1 x[6,3] <= 5
  __wl_bound_00039_lower: + 1 x[6,4] >= 0
  __wl_bound_00039_upper: + 1 x[6,4] <= 7
  __wl_bound_00040_lower: + 1 x[6,5] >= 0
  __wl_bound_00040_upper: + 1 x[6,5] <= 5
  __wl_bound_00041_lower: + 1 x[6,6] >= 0
  __wl_bound_00041_upper: + 1 x[6,6] <= 7
  __wl_bound_00042_lower: + 1 x[6,7] >= 0
  __wl_bound_00042_upper: + 1 x[6,7] <= 6
Bounds
  x[1,1] free
  x[1,2] free
  x[1,3] free
  x[1,4] free
  x[1,5] free
  x[1,6] free
  x[1,7] free
  x[2,1] free
  x[2,2] free
  x[2,3] free
  x[2,4] free
  x[2,5] free
  x[2,6] free
  x[2,7] free
  x[3,1] free
  x[3,2] free
  x[3,3] free
  x[3,4] free
  x[3,5] free
  x[3,6] free
  x[3,7] free
  x[4,1] free
  x[4,2] free
  x[4,3] free
  x[4,4] free
  x[4,5] free
  x[4,6] free
  x[4,7] free
  x[5,1] free
  x[5,2] free
  x[5,3] free
  x[5,4] free
  x[5,5] free
  x[5,6] free
  x[5,7] free
  x[6,1] free
  x[6,2] free
  x[6,3] free
  x[6,4] free
  x[6,5] free
  x[6,6] free
  x[6,7] free
Generals
  x[1,1] x[1,2] x[1,3] x[1,4] x[1,5] x[1,6] x[1,7] x[2,1] x[2,2] x[2,3] x[2,4] x[2,5] x[2,6] x[2,7] x[3,1] x[3,2] x[3,3] x[3,4] x[3,5] x[3,6] x[3,7] x[4,1] x[4,2] x[4,3] x[4,4] x[4,5] x[4,6] x[4,7] x[5,1] x[5,2] x[5,3] x[5,4] x[5,5] x[5,6] x[5,7] x[6,1] x[6,2] x[6,3] x[6,4] x[6,5] x[6,6] x[6,7]
End
