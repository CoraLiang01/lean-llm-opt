\ Canonical explicit-bound formulation for WL comparison.
\ Finite bounds and singleton restrictions merged; original variable domains retained.
Minimize
  obj: + 1091 assign[W11_P00] + 324 assign[W11_P01] + 379 assign[W11_P02] + 272 assign[W11_P03] + 1176 assign[W10_P00] + 1111 assign[W10_P01] + 1380 assign[W10_P02] + 542 assign[W10_P03] + 158 assign[W10_P04] + 922 assign[W10_P05] + 822 assign[W06_P01] + 223 assign[W06_P03] + 1155 assign[W06_P04] + 1055 assign[W06_P05] + 1397 assign[W02_P01] + 953 assign[W02_P02] + 714 assign[W02_P03] + 205 assign[W02_P04] + 1063 assign[W00_P00] + 219 assign[W00_P01] + 1329 assign[W00_P03] + 436 assign[W00_P05] + 642 assign[W04_P00] + 1130 assign[W04_P02] + 133 assign[W04_P03] + 199 assign[W04_P04] + 311 assign[W04_P05] + 102 assign[W12_P00] + 353 assign[W12_P01] + 651 assign[W12_P02] + 102 assign[W12_P03]
Subject To
  R0: + 1 assign[W11_P00] + 1 assign[W10_P00] + 1 assign[W00_P00] + 1 assign[W04_P00] + 1 assign[W12_P00] = 1
  R1: + 1 assign[W11_P01] + 1 assign[W10_P01] + 1 assign[W06_P01] + 1 assign[W02_P01] + 1 assign[W00_P01] + 1 assign[W12_P01] = 1
  R2: + 1 assign[W11_P02] + 1 assign[W10_P02] + 1 assign[W02_P02] + 1 assign[W04_P02] + 1 assign[W12_P02] = 1
  R3: + 1 assign[W11_P03] + 1 assign[W10_P03] + 1 assign[W06_P03] + 1 assign[W02_P03] + 1 assign[W00_P03] + 1 assign[W04_P03] + 1 assign[W12_P03] = 1
  R4: + 1 assign[W10_P04] + 1 assign[W06_P04] + 1 assign[W02_P04] + 1 assign[W04_P04] = 1
  R5: + 1 assign[W10_P05] + 1 assign[W06_P05] + 1 assign[W00_P05] + 1 assign[W04_P05] = 1
  R6: + 1 assign[W12_P00] + 1 assign[W12_P01] + 1 assign[W12_P02] + 1 assign[W12_P03] <= 1
  R7: + 1 assign[W06_P01] + 1 assign[W06_P03] + 1 assign[W06_P04] + 1 assign[W06_P05] <= 1
  R8: + 1 assign[W04_P00] + 1 assign[W04_P02] + 1 assign[W04_P03] + 1 assign[W04_P04] + 1 assign[W04_P05] <= 1
  R9: 0 assign[W11_P00] <= 1
  R10: + 1 assign[W11_P00] + 1 assign[W11_P01] + 1 assign[W11_P02] + 1 assign[W11_P03] <= 1
  R11: + 1 assign[W10_P00] + 1 assign[W10_P01] + 1 assign[W10_P02] + 1 assign[W10_P03] + 1 assign[W10_P04] + 1 assign[W10_P05] <= 1
  R12: + 1 assign[W02_P01] + 1 assign[W02_P02] + 1 assign[W02_P03] + 1 assign[W02_P04] <= 1
  R13: + 1 assign[W00_P00] + 1 assign[W00_P01] + 1 assign[W00_P03] + 1 assign[W00_P05] <= 1
  __wl_bound_00001_lower: + 1 assign[W11_P00] >= 0
  __wl_bound_00001_upper: + 1 assign[W11_P00] <= 1
  __wl_bound_00002_lower: + 1 assign[W11_P01] >= 0
  __wl_bound_00002_upper: + 1 assign[W11_P01] <= 1
  __wl_bound_00003_lower: + 1 assign[W11_P02] >= 0
  __wl_bound_00003_upper: + 1 assign[W11_P02] <= 1
  __wl_bound_00004_lower: + 1 assign[W11_P03] >= 0
  __wl_bound_00004_upper: + 1 assign[W11_P03] <= 1
  __wl_bound_00005_lower: + 1 assign[W10_P00] >= 0
  __wl_bound_00005_upper: + 1 assign[W10_P00] <= 1
  __wl_bound_00006_lower: + 1 assign[W10_P01] >= 0
  __wl_bound_00006_upper: + 1 assign[W10_P01] <= 1
  __wl_bound_00007_lower: + 1 assign[W10_P02] >= 0
  __wl_bound_00007_upper: + 1 assign[W10_P02] <= 1
  __wl_bound_00008_lower: + 1 assign[W10_P03] >= 0
  __wl_bound_00008_upper: + 1 assign[W10_P03] <= 1
  __wl_bound_00009_lower: + 1 assign[W10_P04] >= 0
  __wl_bound_00009_upper: + 1 assign[W10_P04] <= 1
  __wl_bound_00010_lower: + 1 assign[W10_P05] >= 0
  __wl_bound_00010_upper: + 1 assign[W10_P05] <= 1
  __wl_bound_00011_lower: + 1 assign[W06_P01] >= 0
  __wl_bound_00011_upper: + 1 assign[W06_P01] <= 1
  __wl_bound_00012_lower: + 1 assign[W06_P03] >= 0
  __wl_bound_00012_upper: + 1 assign[W06_P03] <= 1
  __wl_bound_00013_lower: + 1 assign[W06_P04] >= 0
  __wl_bound_00013_upper: + 1 assign[W06_P04] <= 1
  __wl_bound_00014_lower: + 1 assign[W06_P05] >= 0
  __wl_bound_00014_upper: + 1 assign[W06_P05] <= 1
  __wl_bound_00015_lower: + 1 assign[W02_P01] >= 0
  __wl_bound_00015_upper: + 1 assign[W02_P01] <= 1
  __wl_bound_00016_lower: + 1 assign[W02_P02] >= 0
  __wl_bound_00016_upper: + 1 assign[W02_P02] <= 1
  __wl_bound_00017_lower: + 1 assign[W02_P03] >= 0
  __wl_bound_00017_upper: + 1 assign[W02_P03] <= 1
  __wl_bound_00018_lower: + 1 assign[W02_P04] >= 0
  __wl_bound_00018_upper: + 1 assign[W02_P04] <= 1
  __wl_bound_00019_lower: + 1 assign[W00_P00] >= 0
  __wl_bound_00019_upper: + 1 assign[W00_P00] <= 1
  __wl_bound_00020_lower: + 1 assign[W00_P01] >= 0
  __wl_bound_00020_upper: + 1 assign[W00_P01] <= 1
  __wl_bound_00021_lower: + 1 assign[W00_P03] >= 0
  __wl_bound_00021_upper: + 1 assign[W00_P03] <= 1
  __wl_bound_00022_lower: + 1 assign[W00_P05] >= 0
  __wl_bound_00022_upper: + 1 assign[W00_P05] <= 1
  __wl_bound_00023_lower: + 1 assign[W04_P00] >= 0
  __wl_bound_00023_upper: + 1 assign[W04_P00] <= 1
  __wl_bound_00024_lower: + 1 assign[W04_P02] >= 0
  __wl_bound_00024_upper: + 1 assign[W04_P02] <= 1
  __wl_bound_00025_lower: + 1 assign[W04_P03] >= 0
  __wl_bound_00025_upper: + 1 assign[W04_P03] <= 1
  __wl_bound_00026_lower: + 1 assign[W04_P04] >= 0
  __wl_bound_00026_upper: + 1 assign[W04_P04] <= 1
  __wl_bound_00027_lower: + 1 assign[W04_P05] >= 0
  __wl_bound_00027_upper: + 1 assign[W04_P05] <= 1
  __wl_bound_00028_lower: + 1 assign[W12_P00] >= 0
  __wl_bound_00028_upper: + 1 assign[W12_P00] <= 1
  __wl_bound_00029_lower: + 1 assign[W12_P01] >= 0
  __wl_bound_00029_upper: + 1 assign[W12_P01] <= 1
  __wl_bound_00030_lower: + 1 assign[W12_P02] >= 0
  __wl_bound_00030_upper: + 1 assign[W12_P02] <= 1
  __wl_bound_00031_lower: + 1 assign[W12_P03] >= 0
  __wl_bound_00031_upper: + 1 assign[W12_P03] <= 1
Bounds
Binaries
  assign[W11_P00] assign[W11_P01] assign[W11_P02] assign[W11_P03] assign[W10_P00] assign[W10_P01] assign[W10_P02] assign[W10_P03] assign[W10_P04] assign[W10_P05] assign[W06_P01] assign[W06_P03] assign[W06_P04] assign[W06_P05] assign[W02_P01] assign[W02_P02] assign[W02_P03] assign[W02_P04] assign[W00_P00] assign[W00_P01] assign[W00_P03] assign[W00_P05] assign[W04_P00] assign[W04_P02] assign[W04_P03] assign[W04_P04] assign[W04_P05] assign[W12_P00] assign[W12_P01] assign[W12_P02] assign[W12_P03]
End
