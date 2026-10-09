\ Canonical explicit-bound formulation for WL comparison.
\ Finite bounds and singleton restrictions merged; original variable domains retained.
Maximize
  obj: + 22 x_G1_O1 + 29 x_G1_O2 + 31 x_G1_O3 + 25 x_G1_O4 + 24 x_G2_O1 + 33 x_G2_O2 + 28 x_G2_O3 + 35 x_G2_O4 + 30 x_G3_O1 + 26 x_G3_O2 + 38 x_G3_O3 + 34 x_G3_O4 + 21 x_G4_O1 + 32 x_G4_O2 + 36 x_G4_O3 + 27 x_G4_O4 + 29 x_G5_O1 + 37 x_G5_O2 + 33 x_G5_O3 + 24 x_G5_O4
Subject To
  choose_G1: + 1 x_G1_O1 + 1 x_G1_O2 + 1 x_G1_O3 + 1 x_G1_O4 = 1
  choose_G2: + 1 x_G2_O1 + 1 x_G2_O2 + 1 x_G2_O3 + 1 x_G2_O4 = 1
  choose_G3: + 1 x_G3_O1 + 1 x_G3_O2 + 1 x_G3_O3 + 1 x_G3_O4 = 1
  choose_G4: + 1 x_G4_O1 + 1 x_G4_O2 + 1 x_G4_O3 + 1 x_G4_O4 = 1
  choose_G5: + 1 x_G5_O1 + 1 x_G5_O2 + 1 x_G5_O3 + 1 x_G5_O4 = 1
  weight_limit: + 6 x_G1_O1 + 9 x_G1_O2 + 10 x_G1_O3 + 7 x_G1_O4 + 7 x_G2_O1 + 11 x_G2_O2 + 8 x_G2_O3 + 12 x_G2_O4 + 9 x_G3_O1 + 7 x_G3_O2 + 13 x_G3_O3 + 11 x_G3_O4 + 5 x_G4_O1 + 10 x_G4_O2 + 12 x_G4_O3 + 8 x_G4_O4 + 8 x_G5_O1 + 12 x_G5_O2 + 10 x_G5_O3 + 6 x_G5_O4 <= 48
  budget_limit: + 8 x_G1_O1 + 11 x_G1_O2 + 13 x_G1_O3 + 9 x_G1_O4 + 8 x_G2_O1 + 13 x_G2_O2 + 10 x_G2_O3 + 14 x_G2_O4 + 12 x_G3_O1 + 9 x_G3_O2 + 16 x_G3_O3 + 13 x_G3_O4 + 7 x_G4_O1 + 12 x_G4_O2 + 15 x_G4_O3 + 10 x_G4_O4 + 11 x_G5_O1 + 15 x_G5_O2 + 12 x_G5_O3 + 8 x_G5_O4 <= 60
  __wl_bound_00001_lower: + 1 x_G1_O1 >= 0
  __wl_bound_00001_upper: + 1 x_G1_O1 <= 1
  __wl_bound_00002_lower: + 1 x_G1_O2 >= 0
  __wl_bound_00002_upper: + 1 x_G1_O2 <= 1
  __wl_bound_00003_lower: + 1 x_G1_O3 >= 0
  __wl_bound_00003_upper: + 1 x_G1_O3 <= 1
  __wl_bound_00004_lower: + 1 x_G1_O4 >= 0
  __wl_bound_00004_upper: + 1 x_G1_O4 <= 1
  __wl_bound_00005_lower: + 1 x_G2_O1 >= 0
  __wl_bound_00005_upper: + 1 x_G2_O1 <= 1
  __wl_bound_00006_lower: + 1 x_G2_O2 >= 0
  __wl_bound_00006_upper: + 1 x_G2_O2 <= 1
  __wl_bound_00007_lower: + 1 x_G2_O3 >= 0
  __wl_bound_00007_upper: + 1 x_G2_O3 <= 1
  __wl_bound_00008_lower: + 1 x_G2_O4 >= 0
  __wl_bound_00008_upper: + 1 x_G2_O4 <= 1
  __wl_bound_00009_lower: + 1 x_G3_O1 >= 0
  __wl_bound_00009_upper: + 1 x_G3_O1 <= 1
  __wl_bound_00010_lower: + 1 x_G3_O2 >= 0
  __wl_bound_00010_upper: + 1 x_G3_O2 <= 1
  __wl_bound_00011_lower: + 1 x_G3_O3 >= 0
  __wl_bound_00011_upper: + 1 x_G3_O3 <= 1
  __wl_bound_00012_lower: + 1 x_G3_O4 >= 0
  __wl_bound_00012_upper: + 1 x_G3_O4 <= 1
  __wl_bound_00013_lower: + 1 x_G4_O1 >= 0
  __wl_bound_00013_upper: + 1 x_G4_O1 <= 1
  __wl_bound_00014_lower: + 1 x_G4_O2 >= 0
  __wl_bound_00014_upper: + 1 x_G4_O2 <= 1
  __wl_bound_00015_lower: + 1 x_G4_O3 >= 0
  __wl_bound_00015_upper: + 1 x_G4_O3 <= 1
  __wl_bound_00016_lower: + 1 x_G4_O4 >= 0
  __wl_bound_00016_upper: + 1 x_G4_O4 <= 1
  __wl_bound_00017_lower: + 1 x_G5_O1 >= 0
  __wl_bound_00017_upper: + 1 x_G5_O1 <= 1
  __wl_bound_00018_lower: + 1 x_G5_O2 >= 0
  __wl_bound_00018_upper: + 1 x_G5_O2 <= 1
  __wl_bound_00019_lower: + 1 x_G5_O3 >= 0
  __wl_bound_00019_upper: + 1 x_G5_O3 <= 1
  __wl_bound_00020_lower: + 1 x_G5_O4 >= 0
  __wl_bound_00020_upper: + 1 x_G5_O4 <= 1
Bounds
Binaries
  x_G1_O1 x_G1_O2 x_G1_O3 x_G1_O4 x_G2_O1 x_G2_O2 x_G2_O3 x_G2_O4 x_G3_O1 x_G3_O2 x_G3_O3 x_G3_O4 x_G4_O1 x_G4_O2 x_G4_O3 x_G4_O4 x_G5_O1 x_G5_O2 x_G5_O3 x_G5_O4
End
