\ Canonical explicit-bound formulation for WL comparison.
\ Finite bounds and singleton restrictions merged; original variable domains retained.
Minimize
  obj: + 3000 y_W1 + 3200 y_W2 + 3100 y_W3 + 2800 y_W4 + 3500 y_W5 + 2700 y_W6 + 2900 y_W7 + 3050 y_W8 + 3100 y_W9 + 2200 y_W10 + 2890 y_W11 + 12 x_W1_S1 + 17 x_W1_S2 + 13 x_W1_S3 + 18 x_W1_S4 + 10 x_W1_S5 + 15 x_W1_S6 + 14 x_W1_S7 + 19 x_W1_S8 + 17 x_W1_S9 + 14 x_W1_S10 + 15 x_W1_S11 + 11 x_W2_S1 + 19 x_W2_S2 + 14 x_W2_S3 + 16 x_W2_S4 + 13 x_W2_S5 + 12 x_W2_S6 + 13 x_W2_S7 + 16 x_W2_S8 + 18 x_W2_S9 + 13 x_W2_S10 + 13 x_W2_S11 + 14 x_W3_S1 + 15 x_W3_S2 + 12 x_W3_S3 + 17 x_W3_S4 + 12 x_W3_S5 + 14 x_W3_S6 + 15 x_W3_S7 + 18 x_W3_S8 + 12 x_W3_S9 + 15 x_W3_S10 + 16 x_W3_S11 + 15 x_W4_S1 + 20 x_W4_S2 + 14 x_W4_S3 + 13 x_W4_S4 + 19 x_W4_S5 + 16 x_W4_S6 + 17 x_W4_S7 + 20 x_W4_S8 + 14 x_W4_S9 + 17 x_W4_S10 + 17 x_W4_S11 + 17 x_W5_S1 + 18 x_W5_S2 + 16 x_W5_S3 + 18 x_W5_S4 + 15 x_W5_S5 + 13 x_W5_S6 + 12 x_W5_S7 + 17 x_W5_S8 + 16 x_W5_S9
    + 16 x_W5_S10 + 11 x_W5_S11 + 13 x_W6_S1 + 14 x_W6_S2 + 15 x_W6_S3 + 17 x_W6_S4 + 11 x_W6_S5 + 17 x_W6_S6 + 13 x_W6_S7 + 19 x_W6_S8 + 15 x_W6_S9 + 18 x_W6_S10 + 13 x_W6_S11 + 12 x_W7_S1 + 17 x_W7_S2 + 11 x_W7_S3 + 14 x_W7_S4 + 12 x_W7_S5 + 16 x_W7_S6 + 14 x_W7_S7 + 16 x_W7_S8 + 14 x_W7_S9 + 14 x_W7_S10 + 14 x_W7_S11 + 16 x_W8_S1 + 15 x_W8_S2 + 14 x_W8_S3 + 19 x_W8_S4 + 14 x_W8_S5 + 16 x_W8_S6 + 15 x_W8_S7 + 18 x_W8_S8 + 17 x_W8_S9 + 19 x_W8_S10 + 15 x_W8_S11 + 16 x_W9_S1 + 13 x_W9_S2 + 16 x_W9_S3 + 16 x_W9_S4 + 12 x_W9_S5 + 14 x_W9_S6 + 12 x_W9_S7 + 15 x_W9_S8 + 21 x_W9_S9 + 15 x_W9_S10 + 19 x_W9_S11 + 14 x_W10_S1 + 15 x_W10_S2 + 18 x_W10_S3 + 13 x_W10_S4 + 15 x_W10_S5 + 18 x_W10_S6 + 16 x_W10_S7 + 15 x_W10_S8 + 15 x_W10_S9 + 17 x_W10_S10 + 21 x_W10_S11 + 15 x_W11_S1 + 16 x_W11_S2 + 17 x_W11_S3 + 18 x_W11_S4 + 17 x_W11_S5
    + 19 x_W11_S6 + 14 x_W11_S7 + 18 x_W11_S8 + 18 x_W11_S9 + 19 x_W11_S10 + 13 x_W11_S11
Subject To
  Demand_S1: + 1 x_W1_S1 + 1 x_W2_S1 + 1 x_W3_S1 + 1 x_W4_S1 + 1 x_W5_S1 + 1 x_W6_S1 + 1 x_W7_S1 + 1 x_W8_S1 + 1 x_W9_S1 + 1 x_W10_S1 + 1 x_W11_S1 = 30
  Demand_S2: + 1 x_W1_S2 + 1 x_W2_S2 + 1 x_W3_S2 + 1 x_W4_S2 + 1 x_W5_S2 + 1 x_W6_S2 + 1 x_W7_S2 + 1 x_W8_S2 + 1 x_W9_S2 + 1 x_W10_S2 + 1 x_W11_S2 = 40
  Demand_S3: + 1 x_W1_S3 + 1 x_W2_S3 + 1 x_W3_S3 + 1 x_W4_S3 + 1 x_W5_S3 + 1 x_W6_S3 + 1 x_W7_S3 + 1 x_W8_S3 + 1 x_W9_S3 + 1 x_W10_S3 + 1 x_W11_S3 = 20
  Demand_S4: + 1 x_W1_S4 + 1 x_W2_S4 + 1 x_W3_S4 + 1 x_W4_S4 + 1 x_W5_S4 + 1 x_W6_S4 + 1 x_W7_S4 + 1 x_W8_S4 + 1 x_W9_S4 + 1 x_W10_S4 + 1 x_W11_S4 = 35
  Demand_S5: + 1 x_W1_S5 + 1 x_W2_S5 + 1 x_W3_S5 + 1 x_W4_S5 + 1 x_W5_S5 + 1 x_W6_S5 + 1 x_W7_S5 + 1 x_W8_S5 + 1 x_W9_S5 + 1 x_W10_S5 + 1 x_W11_S5 = 20
  Demand_S6: + 1 x_W1_S6 + 1 x_W2_S6 + 1 x_W3_S6 + 1 x_W4_S6 + 1 x_W5_S6 + 1 x_W6_S6 + 1 x_W7_S6 + 1 x_W8_S6 + 1 x_W9_S6 + 1 x_W10_S6 + 1 x_W11_S6 = 25
  Demand_S7: + 1 x_W1_S7 + 1 x_W2_S7 + 1 x_W3_S7 + 1 x_W4_S7 + 1 x_W5_S7 + 1 x_W6_S7 + 1 x_W7_S7 + 1 x_W8_S7 + 1 x_W9_S7 + 1 x_W10_S7 + 1 x_W11_S7 = 45
  Demand_S8: + 1 x_W1_S8 + 1 x_W2_S8 + 1 x_W3_S8 + 1 x_W4_S8 + 1 x_W5_S8 + 1 x_W6_S8 + 1 x_W7_S8 + 1 x_W8_S8 + 1 x_W9_S8 + 1 x_W10_S8 + 1 x_W11_S8 = 38
  Demand_S9: + 1 x_W1_S9 + 1 x_W2_S9 + 1 x_W3_S9 + 1 x_W4_S9 + 1 x_W5_S9 + 1 x_W6_S9 + 1 x_W7_S9 + 1 x_W8_S9 + 1 x_W9_S9 + 1 x_W10_S9 + 1 x_W11_S9 = 32
  Demand_S10: + 1 x_W1_S10 + 1 x_W2_S10 + 1 x_W3_S10 + 1 x_W4_S10 + 1 x_W5_S10 + 1 x_W6_S10 + 1 x_W7_S10 + 1 x_W8_S10 + 1 x_W9_S10 + 1 x_W10_S10 + 1 x_W11_S10 = 41
  Demand_S11: + 1 x_W1_S11 + 1 x_W2_S11 + 1 x_W3_S11 + 1 x_W4_S11 + 1 x_W5_S11 + 1 x_W6_S11 + 1 x_W7_S11 + 1 x_W8_S11 + 1 x_W9_S11 + 1 x_W10_S11 + 1 x_W11_S11 = 44
  Capacity_W1: - 180 y_W1 + 1 x_W1_S1 + 1 x_W1_S2 + 1 x_W1_S3 + 1 x_W1_S4 + 1 x_W1_S5 + 1 x_W1_S6 + 1 x_W1_S7 + 1 x_W1_S8 + 1 x_W1_S9 + 1 x_W1_S10 + 1 x_W1_S11 <= 0
  Capacity_W2: - 160 y_W2 + 1 x_W2_S1 + 1 x_W2_S2 + 1 x_W2_S3 + 1 x_W2_S4 + 1 x_W2_S5 + 1 x_W2_S6 + 1 x_W2_S7 + 1 x_W2_S8 + 1 x_W2_S9 + 1 x_W2_S10 + 1 x_W2_S11 <= 0
  Capacity_W3: - 200 y_W3 + 1 x_W3_S1 + 1 x_W3_S2 + 1 x_W3_S3 + 1 x_W3_S4 + 1 x_W3_S5 + 1 x_W3_S6 + 1 x_W3_S7 + 1 x_W3_S8 + 1 x_W3_S9 + 1 x_W3_S10 + 1 x_W3_S11 <= 0
  Capacity_W4: - 150 y_W4 + 1 x_W4_S1 + 1 x_W4_S2 + 1 x_W4_S3 + 1 x_W4_S4 + 1 x_W4_S5 + 1 x_W4_S6 + 1 x_W4_S7 + 1 x_W4_S8 + 1 x_W4_S9 + 1 x_W4_S10 + 1 x_W4_S11 <= 0
  Capacity_W5: - 170 y_W5 + 1 x_W5_S1 + 1 x_W5_S2 + 1 x_W5_S3 + 1 x_W5_S4 + 1 x_W5_S5 + 1 x_W5_S6 + 1 x_W5_S7 + 1 x_W5_S8 + 1 x_W5_S9 + 1 x_W5_S10 + 1 x_W5_S11 <= 0
  Capacity_W6: - 190 y_W6 + 1 x_W6_S1 + 1 x_W6_S2 + 1 x_W6_S3 + 1 x_W6_S4 + 1 x_W6_S5 + 1 x_W6_S6 + 1 x_W6_S7 + 1 x_W6_S8 + 1 x_W6_S9 + 1 x_W6_S10 + 1 x_W6_S11 <= 0
  Capacity_W7: - 160 y_W7 + 1 x_W7_S1 + 1 x_W7_S2 + 1 x_W7_S3 + 1 x_W7_S4 + 1 x_W7_S5 + 1 x_W7_S6 + 1 x_W7_S7 + 1 x_W7_S8 + 1 x_W7_S9 + 1 x_W7_S10 + 1 x_W7_S11 <= 0
  Capacity_W8: - 175 y_W8 + 1 x_W8_S1 + 1 x_W8_S2 + 1 x_W8_S3 + 1 x_W8_S4 + 1 x_W8_S5 + 1 x_W8_S6 + 1 x_W8_S7 + 1 x_W8_S8 + 1 x_W8_S9 + 1 x_W8_S10 + 1 x_W8_S11 <= 0
  Capacity_W9: - 170 y_W9 + 1 x_W9_S1 + 1 x_W9_S2 + 1 x_W9_S3 + 1 x_W9_S4 + 1 x_W9_S5 + 1 x_W9_S6 + 1 x_W9_S7 + 1 x_W9_S8 + 1 x_W9_S9 + 1 x_W9_S10 + 1 x_W9_S11 <= 0
  Capacity_W10: - 180 y_W10 + 1 x_W10_S1 + 1 x_W10_S2 + 1 x_W10_S3 + 1 x_W10_S4 + 1 x_W10_S5 + 1 x_W10_S6 + 1 x_W10_S7 + 1 x_W10_S8 + 1 x_W10_S9 + 1 x_W10_S10 + 1 x_W10_S11 <= 0
  Capacity_W11: - 190 y_W11 + 1 x_W11_S1 + 1 x_W11_S2 + 1 x_W11_S3 + 1 x_W11_S4 + 1 x_W11_S5 + 1 x_W11_S6 + 1 x_W11_S7 + 1 x_W11_S8 + 1 x_W11_S9 + 1 x_W11_S10 + 1 x_W11_S11 <= 0
  __wl_bound_00001_lower: + 1 y_W1 >= 0
  __wl_bound_00001_upper: + 1 y_W1 <= 1
  __wl_bound_00002_lower: + 1 y_W2 >= 0
  __wl_bound_00002_upper: + 1 y_W2 <= 1
  __wl_bound_00003_lower: + 1 y_W3 >= 0
  __wl_bound_00003_upper: + 1 y_W3 <= 1
  __wl_bound_00004_lower: + 1 y_W4 >= 0
  __wl_bound_00004_upper: + 1 y_W4 <= 1
  __wl_bound_00005_lower: + 1 y_W5 >= 0
  __wl_bound_00005_upper: + 1 y_W5 <= 1
  __wl_bound_00006_lower: + 1 y_W6 >= 0
  __wl_bound_00006_upper: + 1 y_W6 <= 1
  __wl_bound_00007_lower: + 1 y_W7 >= 0
  __wl_bound_00007_upper: + 1 y_W7 <= 1
  __wl_bound_00008_lower: + 1 y_W8 >= 0
  __wl_bound_00008_upper: + 1 y_W8 <= 1
  __wl_bound_00009_lower: + 1 y_W9 >= 0
  __wl_bound_00009_upper: + 1 y_W9 <= 1
  __wl_bound_00010_lower: + 1 y_W10 >= 0
  __wl_bound_00010_upper: + 1 y_W10 <= 1
  __wl_bound_00011_lower: + 1 y_W11 >= 0
  __wl_bound_00011_upper: + 1 y_W11 <= 1
  __wl_bound_00012_lower: + 1 x_W1_S1 >= 0
  __wl_bound_00013_lower: + 1 x_W1_S2 >= 0
  __wl_bound_00014_lower: + 1 x_W1_S3 >= 0
  __wl_bound_00015_lower: + 1 x_W1_S4 >= 0
  __wl_bound_00016_lower: + 1 x_W1_S5 >= 0
  __wl_bound_00017_lower: + 1 x_W1_S6 >= 0
  __wl_bound_00018_lower: + 1 x_W1_S7 >= 0
  __wl_bound_00019_lower: + 1 x_W1_S8 >= 0
  __wl_bound_00020_lower: + 1 x_W1_S9 >= 0
  __wl_bound_00021_lower: + 1 x_W1_S10 >= 0
  __wl_bound_00022_lower: + 1 x_W1_S11 >= 0
  __wl_bound_00023_lower: + 1 x_W2_S1 >= 0
  __wl_bound_00024_lower: + 1 x_W2_S2 >= 0
  __wl_bound_00025_lower: + 1 x_W2_S3 >= 0
  __wl_bound_00026_lower: + 1 x_W2_S4 >= 0
  __wl_bound_00027_lower: + 1 x_W2_S5 >= 0
  __wl_bound_00028_lower: + 1 x_W2_S6 >= 0
  __wl_bound_00029_lower: + 1 x_W2_S7 >= 0
  __wl_bound_00030_lower: + 1 x_W2_S8 >= 0
  __wl_bound_00031_lower: + 1 x_W2_S9 >= 0
  __wl_bound_00032_lower: + 1 x_W2_S10 >= 0
  __wl_bound_00033_lower: + 1 x_W2_S11 >= 0
  __wl_bound_00034_lower: + 1 x_W3_S1 >= 0
  __wl_bound_00035_lower: + 1 x_W3_S2 >= 0
  __wl_bound_00036_lower: + 1 x_W3_S3 >= 0
  __wl_bound_00037_lower: + 1 x_W3_S4 >= 0
  __wl_bound_00038_lower: + 1 x_W3_S5 >= 0
  __wl_bound_00039_lower: + 1 x_W3_S6 >= 0
  __wl_bound_00040_lower: + 1 x_W3_S7 >= 0
  __wl_bound_00041_lower: + 1 x_W3_S8 >= 0
  __wl_bound_00042_lower: + 1 x_W3_S9 >= 0
  __wl_bound_00043_lower: + 1 x_W3_S10 >= 0
  __wl_bound_00044_lower: + 1 x_W3_S11 >= 0
  __wl_bound_00045_lower: + 1 x_W4_S1 >= 0
  __wl_bound_00046_lower: + 1 x_W4_S2 >= 0
  __wl_bound_00047_lower: + 1 x_W4_S3 >= 0
  __wl_bound_00048_lower: + 1 x_W4_S4 >= 0
  __wl_bound_00049_lower: + 1 x_W4_S5 >= 0
  __wl_bound_00050_lower: + 1 x_W4_S6 >= 0
  __wl_bound_00051_lower: + 1 x_W4_S7 >= 0
  __wl_bound_00052_lower: + 1 x_W4_S8 >= 0
  __wl_bound_00053_lower: + 1 x_W4_S9 >= 0
  __wl_bound_00054_lower: + 1 x_W4_S10 >= 0
  __wl_bound_00055_lower: + 1 x_W4_S11 >= 0
  __wl_bound_00056_lower: + 1 x_W5_S1 >= 0
  __wl_bound_00057_lower: + 1 x_W5_S2 >= 0
  __wl_bound_00058_lower: + 1 x_W5_S3 >= 0
  __wl_bound_00059_lower: + 1 x_W5_S4 >= 0
  __wl_bound_00060_lower: + 1 x_W5_S5 >= 0
  __wl_bound_00061_lower: + 1 x_W5_S6 >= 0
  __wl_bound_00062_lower: + 1 x_W5_S7 >= 0
  __wl_bound_00063_lower: + 1 x_W5_S8 >= 0
  __wl_bound_00064_lower: + 1 x_W5_S9 >= 0
  __wl_bound_00065_lower: + 1 x_W5_S10 >= 0
  __wl_bound_00066_lower: + 1 x_W5_S11 >= 0
  __wl_bound_00067_lower: + 1 x_W6_S1 >= 0
  __wl_bound_00068_lower: + 1 x_W6_S2 >= 0
  __wl_bound_00069_lower: + 1 x_W6_S3 >= 0
  __wl_bound_00070_lower: + 1 x_W6_S4 >= 0
  __wl_bound_00071_lower: + 1 x_W6_S5 >= 0
  __wl_bound_00072_lower: + 1 x_W6_S6 >= 0
  __wl_bound_00073_lower: + 1 x_W6_S7 >= 0
  __wl_bound_00074_lower: + 1 x_W6_S8 >= 0
  __wl_bound_00075_lower: + 1 x_W6_S9 >= 0
  __wl_bound_00076_lower: + 1 x_W6_S10 >= 0
  __wl_bound_00077_lower: + 1 x_W6_S11 >= 0
  __wl_bound_00078_lower: + 1 x_W7_S1 >= 0
  __wl_bound_00079_lower: + 1 x_W7_S2 >= 0
  __wl_bound_00080_lower: + 1 x_W7_S3 >= 0
  __wl_bound_00081_lower: + 1 x_W7_S4 >= 0
  __wl_bound_00082_lower: + 1 x_W7_S5 >= 0
  __wl_bound_00083_lower: + 1 x_W7_S6 >= 0
  __wl_bound_00084_lower: + 1 x_W7_S7 >= 0
  __wl_bound_00085_lower: + 1 x_W7_S8 >= 0
  __wl_bound_00086_lower: + 1 x_W7_S9 >= 0
  __wl_bound_00087_lower: + 1 x_W7_S10 >= 0
  __wl_bound_00088_lower: + 1 x_W7_S11 >= 0
  __wl_bound_00089_lower: + 1 x_W8_S1 >= 0
  __wl_bound_00090_lower: + 1 x_W8_S2 >= 0
  __wl_bound_00091_lower: + 1 x_W8_S3 >= 0
  __wl_bound_00092_lower: + 1 x_W8_S4 >= 0
  __wl_bound_00093_lower: + 1 x_W8_S5 >= 0
  __wl_bound_00094_lower: + 1 x_W8_S6 >= 0
  __wl_bound_00095_lower: + 1 x_W8_S7 >= 0
  __wl_bound_00096_lower: + 1 x_W8_S8 >= 0
  __wl_bound_00097_lower: + 1 x_W8_S9 >= 0
  __wl_bound_00098_lower: + 1 x_W8_S10 >= 0
  __wl_bound_00099_lower: + 1 x_W8_S11 >= 0
  __wl_bound_00100_lower: + 1 x_W9_S1 >= 0
  __wl_bound_00101_lower: + 1 x_W9_S2 >= 0
  __wl_bound_00102_lower: + 1 x_W9_S3 >= 0
  __wl_bound_00103_lower: + 1 x_W9_S4 >= 0
  __wl_bound_00104_lower: + 1 x_W9_S5 >= 0
  __wl_bound_00105_lower: + 1 x_W9_S6 >= 0
  __wl_bound_00106_lower: + 1 x_W9_S7 >= 0
  __wl_bound_00107_lower: + 1 x_W9_S8 >= 0
  __wl_bound_00108_lower: + 1 x_W9_S9 >= 0
  __wl_bound_00109_lower: + 1 x_W9_S10 >= 0
  __wl_bound_00110_lower: + 1 x_W9_S11 >= 0
  __wl_bound_00111_lower: + 1 x_W10_S1 >= 0
  __wl_bound_00112_lower: + 1 x_W10_S2 >= 0
  __wl_bound_00113_lower: + 1 x_W10_S3 >= 0
  __wl_bound_00114_lower: + 1 x_W10_S4 >= 0
  __wl_bound_00115_lower: + 1 x_W10_S5 >= 0
  __wl_bound_00116_lower: + 1 x_W10_S6 >= 0
  __wl_bound_00117_lower: + 1 x_W10_S7 >= 0
  __wl_bound_00118_lower: + 1 x_W10_S8 >= 0
  __wl_bound_00119_lower: + 1 x_W10_S9 >= 0
  __wl_bound_00120_lower: + 1 x_W10_S10 >= 0
  __wl_bound_00121_lower: + 1 x_W10_S11 >= 0
  __wl_bound_00122_lower: + 1 x_W11_S1 >= 0
  __wl_bound_00123_lower: + 1 x_W11_S2 >= 0
  __wl_bound_00124_lower: + 1 x_W11_S3 >= 0
  __wl_bound_00125_lower: + 1 x_W11_S4 >= 0
  __wl_bound_00126_lower: + 1 x_W11_S5 >= 0
  __wl_bound_00127_lower: + 1 x_W11_S6 >= 0
  __wl_bound_00128_lower: + 1 x_W11_S7 >= 0
  __wl_bound_00129_lower: + 1 x_W11_S8 >= 0
  __wl_bound_00130_lower: + 1 x_W11_S9 >= 0
  __wl_bound_00131_lower: + 1 x_W11_S10 >= 0
  __wl_bound_00132_lower: + 1 x_W11_S11 >= 0
Bounds
  x_W1_S1 free
  x_W1_S2 free
  x_W1_S3 free
  x_W1_S4 free
  x_W1_S5 free
  x_W1_S6 free
  x_W1_S7 free
  x_W1_S8 free
  x_W1_S9 free
  x_W1_S10 free
  x_W1_S11 free
  x_W2_S1 free
  x_W2_S2 free
  x_W2_S3 free
  x_W2_S4 free
  x_W2_S5 free
  x_W2_S6 free
  x_W2_S7 free
  x_W2_S8 free
  x_W2_S9 free
  x_W2_S10 free
  x_W2_S11 free
  x_W3_S1 free
  x_W3_S2 free
  x_W3_S3 free
  x_W3_S4 free
  x_W3_S5 free
  x_W3_S6 free
  x_W3_S7 free
  x_W3_S8 free
  x_W3_S9 free
  x_W3_S10 free
  x_W3_S11 free
  x_W4_S1 free
  x_W4_S2 free
  x_W4_S3 free
  x_W4_S4 free
  x_W4_S5 free
  x_W4_S6 free
  x_W4_S7 free
  x_W4_S8 free
  x_W4_S9 free
  x_W4_S10 free
  x_W4_S11 free
  x_W5_S1 free
  x_W5_S2 free
  x_W5_S3 free
  x_W5_S4 free
  x_W5_S5 free
  x_W5_S6 free
  x_W5_S7 free
  x_W5_S8 free
  x_W5_S9 free
  x_W5_S10 free
  x_W5_S11 free
  x_W6_S1 free
  x_W6_S2 free
  x_W6_S3 free
  x_W6_S4 free
  x_W6_S5 free
  x_W6_S6 free
  x_W6_S7 free
  x_W6_S8 free
  x_W6_S9 free
  x_W6_S10 free
  x_W6_S11 free
  x_W7_S1 free
  x_W7_S2 free
  x_W7_S3 free
  x_W7_S4 free
  x_W7_S5 free
  x_W7_S6 free
  x_W7_S7 free
  x_W7_S8 free
  x_W7_S9 free
  x_W7_S10 free
  x_W7_S11 free
  x_W8_S1 free
  x_W8_S2 free
  x_W8_S3 free
  x_W8_S4 free
  x_W8_S5 free
  x_W8_S6 free
  x_W8_S7 free
  x_W8_S8 free
  x_W8_S9 free
  x_W8_S10 free
  x_W8_S11 free
  x_W9_S1 free
  x_W9_S2 free
  x_W9_S3 free
  x_W9_S4 free
  x_W9_S5 free
  x_W9_S6 free
  x_W9_S7 free
  x_W9_S8 free
  x_W9_S9 free
  x_W9_S10 free
  x_W9_S11 free
  x_W10_S1 free
  x_W10_S2 free
  x_W10_S3 free
  x_W10_S4 free
  x_W10_S5 free
  x_W10_S6 free
  x_W10_S7 free
  x_W10_S8 free
  x_W10_S9 free
  x_W10_S10 free
  x_W10_S11 free
  x_W11_S1 free
  x_W11_S2 free
  x_W11_S3 free
  x_W11_S4 free
  x_W11_S5 free
  x_W11_S6 free
  x_W11_S7 free
  x_W11_S8 free
  x_W11_S9 free
  x_W11_S10 free
  x_W11_S11 free
Binaries
  y_W1 y_W2 y_W3 y_W4 y_W5 y_W6 y_W7 y_W8 y_W9 y_W10 y_W11
End
