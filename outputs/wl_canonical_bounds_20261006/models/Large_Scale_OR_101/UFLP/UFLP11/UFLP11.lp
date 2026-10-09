\ Canonical explicit-bound formulation for WL comparison.
\ Finite bounds and singleton restrictions merged; original variable domains retained.
Minimize
  obj: + 175 y[A2] + 300 y[A3] + 375 y[A4] + 500 y[A5] + 200 y[A6] + 260 y[A7] + 220 y[A8] + 320 y[A9] + 280 y[A10] + 350 y[A11] + 420 y[A12] + 470 y[A13] + 520 y[A14] + 560 y[A15] + 8 x[A1,B1] + 4 x[A1,B2] + 3 x[A1,B3] + 6 x[A1,B4] + 7 x[A1,B5] + 5 x[A1,B6] + 9 x[A1,B7] + 8 x[A1,B8] + 5 x[A2,B1] + 2 x[A2,B2] + 3 x[A2,B3] + 5 x[A2,B4] + 6 x[A2,B5] + 4 x[A2,B6] + 7 x[A2,B7] + 6 x[A2,B8] + 4 x[A3,B1] + 3 x[A3,B2] + 4 x[A3,B3] + 6 x[A3,B4] + 5 x[A3,B5] + 5 x[A3,B6] + 6 x[A3,B7] + 7 x[A3,B8] + 9 x[A4,B1] + 7 x[A4,B2] + 5 x[A4,B3] + 8 x[A4,B4] + 9 x[A4,B5] + 6 x[A4,B6] + 10 x[A4,B7] + 7 x[A4,B8] + 10 x[A5,B1] + 4 x[A5,B2] + 2 x[A5,B3] + 6 x[A5,B4] + 8 x[A5,B5] + 5 x[A5,B6] + 7 x[A5,B7] + 3 x[A5,B8] + 6 x[A6,B1] + 5 x[A6,B2] + 4 x[A6,B3] + 5 x[A6,B4] + 7 x[A6,B5] + 6 x[A6,B6] + 8 x[A6,B7] + 5 x[A6,B8] + 7 x[A7,B1] + 6 x[A7,B2] + 5 x[A7,B3]
    + 4 x[A7,B4] + 6 x[A7,B5] + 7 x[A7,B6] + 9 x[A7,B7] + 6 x[A7,B8] + 5 x[A8,B1] + 4 x[A8,B2] + 6 x[A8,B3] + 3 x[A8,B4] + 5 x[A8,B5] + 6 x[A8,B6] + 7 x[A8,B7] + 6 x[A8,B8] + 8 x[A9,B1] + 7 x[A9,B2] + 6 x[A9,B3] + 7 x[A9,B4] + 9 x[A9,B5] + 8 x[A9,B6] + 10 x[A9,B7] + 7 x[A9,B8] + 6 x[A10,B1] + 5 x[A10,B2] + 7 x[A10,B3] + 4 x[A10,B4] + 6 x[A10,B5] + 5 x[A10,B6] + 7 x[A10,B7] + 5 x[A10,B8] + 9 x[A11,B1] + 6 x[A11,B2] + 4 x[A11,B3] + 6 x[A11,B4] + 8 x[A11,B5] + 7 x[A11,B6] + 9 x[A11,B7] + 6 x[A11,B8] + 7 x[A12,B1] + 5 x[A12,B2] + 6 x[A12,B3] + 5 x[A12,B4] + 6 x[A12,B5] + 5 x[A12,B6] + 8 x[A12,B7] + 5 x[A12,B8] + 8 x[A13,B1] + 6 x[A13,B2] + 5 x[A13,B3] + 6 x[A13,B4] + 7 x[A13,B5] + 6 x[A13,B6] + 8 x[A13,B7] + 7 x[A13,B8] + 9 x[A14,B1] + 5 x[A14,B2] + 3 x[A14,B3] + 5 x[A14,B4] + 7 x[A14,B5] + 4 x[A14,B6] + 6 x[A14,B7] + 4 x[A14,B8]
    + 10 x[A15,B1] + 6 x[A15,B2] + 4 x[A15,B3] + 5 x[A15,B4] + 8 x[A15,B5] + 5 x[A15,B6] + 7 x[A15,B7] + 5 x[A15,B8]
Subject To
  Demand_B1: + 1 x[A1,B1] + 1 x[A2,B1] + 1 x[A3,B1] + 1 x[A4,B1] + 1 x[A5,B1] + 1 x[A6,B1] + 1 x[A7,B1] + 1 x[A8,B1] + 1 x[A9,B1] + 1 x[A10,B1] + 1 x[A11,B1] + 1 x[A12,B1] + 1 x[A13,B1] + 1 x[A14,B1] + 1 x[A15,B1] = 30
  Demand_B2: + 1 x[A1,B2] + 1 x[A2,B2] + 1 x[A3,B2] + 1 x[A4,B2] + 1 x[A5,B2] + 1 x[A6,B2] + 1 x[A7,B2] + 1 x[A8,B2] + 1 x[A9,B2] + 1 x[A10,B2] + 1 x[A11,B2] + 1 x[A12,B2] + 1 x[A13,B2] + 1 x[A14,B2] + 1 x[A15,B2] = 25
  Demand_B3: + 1 x[A1,B3] + 1 x[A2,B3] + 1 x[A3,B3] + 1 x[A4,B3] + 1 x[A5,B3] + 1 x[A6,B3] + 1 x[A7,B3] + 1 x[A8,B3] + 1 x[A9,B3] + 1 x[A10,B3] + 1 x[A11,B3] + 1 x[A12,B3] + 1 x[A13,B3] + 1 x[A14,B3] + 1 x[A15,B3] = 20
  Demand_B4: + 1 x[A1,B4] + 1 x[A2,B4] + 1 x[A3,B4] + 1 x[A4,B4] + 1 x[A5,B4] + 1 x[A6,B4] + 1 x[A7,B4] + 1 x[A8,B4] + 1 x[A9,B4] + 1 x[A10,B4] + 1 x[A11,B4] + 1 x[A12,B4] + 1 x[A13,B4] + 1 x[A14,B4] + 1 x[A15,B4] = 35
  Demand_B5: + 1 x[A1,B5] + 1 x[A2,B5] + 1 x[A3,B5] + 1 x[A4,B5] + 1 x[A5,B5] + 1 x[A6,B5] + 1 x[A7,B5] + 1 x[A8,B5] + 1 x[A9,B5] + 1 x[A10,B5] + 1 x[A11,B5] + 1 x[A12,B5] + 1 x[A13,B5] + 1 x[A14,B5] + 1 x[A15,B5] = 25
  Demand_B6: + 1 x[A1,B6] + 1 x[A2,B6] + 1 x[A3,B6] + 1 x[A4,B6] + 1 x[A5,B6] + 1 x[A6,B6] + 1 x[A7,B6] + 1 x[A8,B6] + 1 x[A9,B6] + 1 x[A10,B6] + 1 x[A11,B6] + 1 x[A12,B6] + 1 x[A13,B6] + 1 x[A14,B6] + 1 x[A15,B6] = 30
  Demand_B7: + 1 x[A1,B7] + 1 x[A2,B7] + 1 x[A3,B7] + 1 x[A4,B7] + 1 x[A5,B7] + 1 x[A6,B7] + 1 x[A7,B7] + 1 x[A8,B7] + 1 x[A9,B7] + 1 x[A10,B7] + 1 x[A11,B7] + 1 x[A12,B7] + 1 x[A13,B7] + 1 x[A14,B7] + 1 x[A15,B7] = 25
  Demand_B8: + 1 x[A1,B8] + 1 x[A2,B8] + 1 x[A3,B8] + 1 x[A4,B8] + 1 x[A5,B8] + 1 x[A6,B8] + 1 x[A7,B8] + 1 x[A8,B8] + 1 x[A9,B8] + 1 x[A10,B8] + 1 x[A11,B8] + 1 x[A12,B8] + 1 x[A13,B8] + 1 x[A14,B8] + 1 x[A15,B8] = 30
  Capacity_A1: + 1 x[A1,B1] + 1 x[A1,B2] + 1 x[A1,B3] + 1 x[A1,B4] + 1 x[A1,B5] + 1 x[A1,B6] + 1 x[A1,B7] + 1 x[A1,B8] - 30 y[A1] <= 0
  Capacity_A2: - 10 y[A2] + 1 x[A2,B1] + 1 x[A2,B2] + 1 x[A2,B3] + 1 x[A2,B4] + 1 x[A2,B5] + 1 x[A2,B6] + 1 x[A2,B7] + 1 x[A2,B8] <= 0
  Capacity_A3: - 20 y[A3] + 1 x[A3,B1] + 1 x[A3,B2] + 1 x[A3,B3] + 1 x[A3,B4] + 1 x[A3,B5] + 1 x[A3,B6] + 1 x[A3,B7] + 1 x[A3,B8] <= 0
  Capacity_A4: - 30 y[A4] + 1 x[A4,B1] + 1 x[A4,B2] + 1 x[A4,B3] + 1 x[A4,B4] + 1 x[A4,B5] + 1 x[A4,B6] + 1 x[A4,B7] + 1 x[A4,B8] <= 0
  Capacity_A5: - 40 y[A5] + 1 x[A5,B1] + 1 x[A5,B2] + 1 x[A5,B3] + 1 x[A5,B4] + 1 x[A5,B5] + 1 x[A5,B6] + 1 x[A5,B7] + 1 x[A5,B8] <= 0
  Capacity_A6: - 20 y[A6] + 1 x[A6,B1] + 1 x[A6,B2] + 1 x[A6,B3] + 1 x[A6,B4] + 1 x[A6,B5] + 1 x[A6,B6] + 1 x[A6,B7] + 1 x[A6,B8] <= 0
  Capacity_A7: - 25 y[A7] + 1 x[A7,B1] + 1 x[A7,B2] + 1 x[A7,B3] + 1 x[A7,B4] + 1 x[A7,B5] + 1 x[A7,B6] + 1 x[A7,B7] + 1 x[A7,B8] <= 0
  Capacity_A8: - 30 y[A8] + 1 x[A8,B1] + 1 x[A8,B2] + 1 x[A8,B3] + 1 x[A8,B4] + 1 x[A8,B5] + 1 x[A8,B6] + 1 x[A8,B7] + 1 x[A8,B8] <= 0
  Capacity_A9: - 35 y[A9] + 1 x[A9,B1] + 1 x[A9,B2] + 1 x[A9,B3] + 1 x[A9,B4] + 1 x[A9,B5] + 1 x[A9,B6] + 1 x[A9,B7] + 1 x[A9,B8] <= 0
  Capacity_A10: - 20 y[A10] + 1 x[A10,B1] + 1 x[A10,B2] + 1 x[A10,B3] + 1 x[A10,B4] + 1 x[A10,B5] + 1 x[A10,B6] + 1 x[A10,B7] + 1 x[A10,B8] <= 0
  Capacity_A11: - 40 y[A11] + 1 x[A11,B1] + 1 x[A11,B2] + 1 x[A11,B3] + 1 x[A11,B4] + 1 x[A11,B5] + 1 x[A11,B6] + 1 x[A11,B7] + 1 x[A11,B8] <= 0
  Capacity_A12: - 25 y[A12] + 1 x[A12,B1] + 1 x[A12,B2] + 1 x[A12,B3] + 1 x[A12,B4] + 1 x[A12,B5] + 1 x[A12,B6] + 1 x[A12,B7] + 1 x[A12,B8] <= 0
  Capacity_A13: - 30 y[A13] + 1 x[A13,B1] + 1 x[A13,B2] + 1 x[A13,B3] + 1 x[A13,B4] + 1 x[A13,B5] + 1 x[A13,B6] + 1 x[A13,B7] + 1 x[A13,B8] <= 0
  Capacity_A14: - 50 y[A14] + 1 x[A14,B1] + 1 x[A14,B2] + 1 x[A14,B3] + 1 x[A14,B4] + 1 x[A14,B5] + 1 x[A14,B6] + 1 x[A14,B7] + 1 x[A14,B8] <= 0
  Capacity_A15: - 45 y[A15] + 1 x[A15,B1] + 1 x[A15,B2] + 1 x[A15,B3] + 1 x[A15,B4] + 1 x[A15,B5] + 1 x[A15,B6] + 1 x[A15,B7] + 1 x[A15,B8] <= 0
  __wl_bound_00001_lower: + 1 y[A2] >= 0
  __wl_bound_00001_upper: + 1 y[A2] <= 1
  __wl_bound_00002_lower: + 1 y[A3] >= 0
  __wl_bound_00002_upper: + 1 y[A3] <= 1
  __wl_bound_00003_lower: + 1 y[A4] >= 0
  __wl_bound_00003_upper: + 1 y[A4] <= 1
  __wl_bound_00004_lower: + 1 y[A5] >= 0
  __wl_bound_00004_upper: + 1 y[A5] <= 1
  __wl_bound_00005_lower: + 1 y[A6] >= 0
  __wl_bound_00005_upper: + 1 y[A6] <= 1
  __wl_bound_00006_lower: + 1 y[A7] >= 0
  __wl_bound_00006_upper: + 1 y[A7] <= 1
  __wl_bound_00007_lower: + 1 y[A8] >= 0
  __wl_bound_00007_upper: + 1 y[A8] <= 1
  __wl_bound_00008_lower: + 1 y[A9] >= 0
  __wl_bound_00008_upper: + 1 y[A9] <= 1
  __wl_bound_00009_lower: + 1 y[A10] >= 0
  __wl_bound_00009_upper: + 1 y[A10] <= 1
  __wl_bound_00010_lower: + 1 y[A11] >= 0
  __wl_bound_00010_upper: + 1 y[A11] <= 1
  __wl_bound_00011_lower: + 1 y[A12] >= 0
  __wl_bound_00011_upper: + 1 y[A12] <= 1
  __wl_bound_00012_lower: + 1 y[A13] >= 0
  __wl_bound_00012_upper: + 1 y[A13] <= 1
  __wl_bound_00013_lower: + 1 y[A14] >= 0
  __wl_bound_00013_upper: + 1 y[A14] <= 1
  __wl_bound_00014_lower: + 1 y[A15] >= 0
  __wl_bound_00014_upper: + 1 y[A15] <= 1
  __wl_bound_00015_lower: + 1 x[A1,B1] >= 0
  __wl_bound_00016_lower: + 1 x[A1,B2] >= 0
  __wl_bound_00017_lower: + 1 x[A1,B3] >= 0
  __wl_bound_00018_lower: + 1 x[A1,B4] >= 0
  __wl_bound_00019_lower: + 1 x[A1,B5] >= 0
  __wl_bound_00020_lower: + 1 x[A1,B6] >= 0
  __wl_bound_00021_lower: + 1 x[A1,B7] >= 0
  __wl_bound_00022_lower: + 1 x[A1,B8] >= 0
  __wl_bound_00023_lower: + 1 x[A2,B1] >= 0
  __wl_bound_00024_lower: + 1 x[A2,B2] >= 0
  __wl_bound_00025_lower: + 1 x[A2,B3] >= 0
  __wl_bound_00026_lower: + 1 x[A2,B4] >= 0
  __wl_bound_00027_lower: + 1 x[A2,B5] >= 0
  __wl_bound_00028_lower: + 1 x[A2,B6] >= 0
  __wl_bound_00029_lower: + 1 x[A2,B7] >= 0
  __wl_bound_00030_lower: + 1 x[A2,B8] >= 0
  __wl_bound_00031_lower: + 1 x[A3,B1] >= 0
  __wl_bound_00032_lower: + 1 x[A3,B2] >= 0
  __wl_bound_00033_lower: + 1 x[A3,B3] >= 0
  __wl_bound_00034_lower: + 1 x[A3,B4] >= 0
  __wl_bound_00035_lower: + 1 x[A3,B5] >= 0
  __wl_bound_00036_lower: + 1 x[A3,B6] >= 0
  __wl_bound_00037_lower: + 1 x[A3,B7] >= 0
  __wl_bound_00038_lower: + 1 x[A3,B8] >= 0
  __wl_bound_00039_lower: + 1 x[A4,B1] >= 0
  __wl_bound_00040_lower: + 1 x[A4,B2] >= 0
  __wl_bound_00041_lower: + 1 x[A4,B3] >= 0
  __wl_bound_00042_lower: + 1 x[A4,B4] >= 0
  __wl_bound_00043_lower: + 1 x[A4,B5] >= 0
  __wl_bound_00044_lower: + 1 x[A4,B6] >= 0
  __wl_bound_00045_lower: + 1 x[A4,B7] >= 0
  __wl_bound_00046_lower: + 1 x[A4,B8] >= 0
  __wl_bound_00047_lower: + 1 x[A5,B1] >= 0
  __wl_bound_00048_lower: + 1 x[A5,B2] >= 0
  __wl_bound_00049_lower: + 1 x[A5,B3] >= 0
  __wl_bound_00050_lower: + 1 x[A5,B4] >= 0
  __wl_bound_00051_lower: + 1 x[A5,B5] >= 0
  __wl_bound_00052_lower: + 1 x[A5,B6] >= 0
  __wl_bound_00053_lower: + 1 x[A5,B7] >= 0
  __wl_bound_00054_lower: + 1 x[A5,B8] >= 0
  __wl_bound_00055_lower: + 1 x[A6,B1] >= 0
  __wl_bound_00056_lower: + 1 x[A6,B2] >= 0
  __wl_bound_00057_lower: + 1 x[A6,B3] >= 0
  __wl_bound_00058_lower: + 1 x[A6,B4] >= 0
  __wl_bound_00059_lower: + 1 x[A6,B5] >= 0
  __wl_bound_00060_lower: + 1 x[A6,B6] >= 0
  __wl_bound_00061_lower: + 1 x[A6,B7] >= 0
  __wl_bound_00062_lower: + 1 x[A6,B8] >= 0
  __wl_bound_00063_lower: + 1 x[A7,B1] >= 0
  __wl_bound_00064_lower: + 1 x[A7,B2] >= 0
  __wl_bound_00065_lower: + 1 x[A7,B3] >= 0
  __wl_bound_00066_lower: + 1 x[A7,B4] >= 0
  __wl_bound_00067_lower: + 1 x[A7,B5] >= 0
  __wl_bound_00068_lower: + 1 x[A7,B6] >= 0
  __wl_bound_00069_lower: + 1 x[A7,B7] >= 0
  __wl_bound_00070_lower: + 1 x[A7,B8] >= 0
  __wl_bound_00071_lower: + 1 x[A8,B1] >= 0
  __wl_bound_00072_lower: + 1 x[A8,B2] >= 0
  __wl_bound_00073_lower: + 1 x[A8,B3] >= 0
  __wl_bound_00074_lower: + 1 x[A8,B4] >= 0
  __wl_bound_00075_lower: + 1 x[A8,B5] >= 0
  __wl_bound_00076_lower: + 1 x[A8,B6] >= 0
  __wl_bound_00077_lower: + 1 x[A8,B7] >= 0
  __wl_bound_00078_lower: + 1 x[A8,B8] >= 0
  __wl_bound_00079_lower: + 1 x[A9,B1] >= 0
  __wl_bound_00080_lower: + 1 x[A9,B2] >= 0
  __wl_bound_00081_lower: + 1 x[A9,B3] >= 0
  __wl_bound_00082_lower: + 1 x[A9,B4] >= 0
  __wl_bound_00083_lower: + 1 x[A9,B5] >= 0
  __wl_bound_00084_lower: + 1 x[A9,B6] >= 0
  __wl_bound_00085_lower: + 1 x[A9,B7] >= 0
  __wl_bound_00086_lower: + 1 x[A9,B8] >= 0
  __wl_bound_00087_lower: + 1 x[A10,B1] >= 0
  __wl_bound_00088_lower: + 1 x[A10,B2] >= 0
  __wl_bound_00089_lower: + 1 x[A10,B3] >= 0
  __wl_bound_00090_lower: + 1 x[A10,B4] >= 0
  __wl_bound_00091_lower: + 1 x[A10,B5] >= 0
  __wl_bound_00092_lower: + 1 x[A10,B6] >= 0
  __wl_bound_00093_lower: + 1 x[A10,B7] >= 0
  __wl_bound_00094_lower: + 1 x[A10,B8] >= 0
  __wl_bound_00095_lower: + 1 x[A11,B1] >= 0
  __wl_bound_00096_lower: + 1 x[A11,B2] >= 0
  __wl_bound_00097_lower: + 1 x[A11,B3] >= 0
  __wl_bound_00098_lower: + 1 x[A11,B4] >= 0
  __wl_bound_00099_lower: + 1 x[A11,B5] >= 0
  __wl_bound_00100_lower: + 1 x[A11,B6] >= 0
  __wl_bound_00101_lower: + 1 x[A11,B7] >= 0
  __wl_bound_00102_lower: + 1 x[A11,B8] >= 0
  __wl_bound_00103_lower: + 1 x[A12,B1] >= 0
  __wl_bound_00104_lower: + 1 x[A12,B2] >= 0
  __wl_bound_00105_lower: + 1 x[A12,B3] >= 0
  __wl_bound_00106_lower: + 1 x[A12,B4] >= 0
  __wl_bound_00107_lower: + 1 x[A12,B5] >= 0
  __wl_bound_00108_lower: + 1 x[A12,B6] >= 0
  __wl_bound_00109_lower: + 1 x[A12,B7] >= 0
  __wl_bound_00110_lower: + 1 x[A12,B8] >= 0
  __wl_bound_00111_lower: + 1 x[A13,B1] >= 0
  __wl_bound_00112_lower: + 1 x[A13,B2] >= 0
  __wl_bound_00113_lower: + 1 x[A13,B3] >= 0
  __wl_bound_00114_lower: + 1 x[A13,B4] >= 0
  __wl_bound_00115_lower: + 1 x[A13,B5] >= 0
  __wl_bound_00116_lower: + 1 x[A13,B6] >= 0
  __wl_bound_00117_lower: + 1 x[A13,B7] >= 0
  __wl_bound_00118_lower: + 1 x[A13,B8] >= 0
  __wl_bound_00119_lower: + 1 x[A14,B1] >= 0
  __wl_bound_00120_lower: + 1 x[A14,B2] >= 0
  __wl_bound_00121_lower: + 1 x[A14,B3] >= 0
  __wl_bound_00122_lower: + 1 x[A14,B4] >= 0
  __wl_bound_00123_lower: + 1 x[A14,B5] >= 0
  __wl_bound_00124_lower: + 1 x[A14,B6] >= 0
  __wl_bound_00125_lower: + 1 x[A14,B7] >= 0
  __wl_bound_00126_lower: + 1 x[A14,B8] >= 0
  __wl_bound_00127_lower: + 1 x[A15,B1] >= 0
  __wl_bound_00128_lower: + 1 x[A15,B2] >= 0
  __wl_bound_00129_lower: + 1 x[A15,B3] >= 0
  __wl_bound_00130_lower: + 1 x[A15,B4] >= 0
  __wl_bound_00131_lower: + 1 x[A15,B5] >= 0
  __wl_bound_00132_lower: + 1 x[A15,B6] >= 0
  __wl_bound_00133_lower: + 1 x[A15,B7] >= 0
  __wl_bound_00134_lower: + 1 x[A15,B8] >= 0
  __wl_bound_00135_lower: + 1 y[A1] >= 0
  __wl_bound_00135_upper: + 1 y[A1] <= 1
Bounds
  x[A1,B1] free
  x[A1,B2] free
  x[A1,B3] free
  x[A1,B4] free
  x[A1,B5] free
  x[A1,B6] free
  x[A1,B7] free
  x[A1,B8] free
  x[A2,B1] free
  x[A2,B2] free
  x[A2,B3] free
  x[A2,B4] free
  x[A2,B5] free
  x[A2,B6] free
  x[A2,B7] free
  x[A2,B8] free
  x[A3,B1] free
  x[A3,B2] free
  x[A3,B3] free
  x[A3,B4] free
  x[A3,B5] free
  x[A3,B6] free
  x[A3,B7] free
  x[A3,B8] free
  x[A4,B1] free
  x[A4,B2] free
  x[A4,B3] free
  x[A4,B4] free
  x[A4,B5] free
  x[A4,B6] free
  x[A4,B7] free
  x[A4,B8] free
  x[A5,B1] free
  x[A5,B2] free
  x[A5,B3] free
  x[A5,B4] free
  x[A5,B5] free
  x[A5,B6] free
  x[A5,B7] free
  x[A5,B8] free
  x[A6,B1] free
  x[A6,B2] free
  x[A6,B3] free
  x[A6,B4] free
  x[A6,B5] free
  x[A6,B6] free
  x[A6,B7] free
  x[A6,B8] free
  x[A7,B1] free
  x[A7,B2] free
  x[A7,B3] free
  x[A7,B4] free
  x[A7,B5] free
  x[A7,B6] free
  x[A7,B7] free
  x[A7,B8] free
  x[A8,B1] free
  x[A8,B2] free
  x[A8,B3] free
  x[A8,B4] free
  x[A8,B5] free
  x[A8,B6] free
  x[A8,B7] free
  x[A8,B8] free
  x[A9,B1] free
  x[A9,B2] free
  x[A9,B3] free
  x[A9,B4] free
  x[A9,B5] free
  x[A9,B6] free
  x[A9,B7] free
  x[A9,B8] free
  x[A10,B1] free
  x[A10,B2] free
  x[A10,B3] free
  x[A10,B4] free
  x[A10,B5] free
  x[A10,B6] free
  x[A10,B7] free
  x[A10,B8] free
  x[A11,B1] free
  x[A11,B2] free
  x[A11,B3] free
  x[A11,B4] free
  x[A11,B5] free
  x[A11,B6] free
  x[A11,B7] free
  x[A11,B8] free
  x[A12,B1] free
  x[A12,B2] free
  x[A12,B3] free
  x[A12,B4] free
  x[A12,B5] free
  x[A12,B6] free
  x[A12,B7] free
  x[A12,B8] free
  x[A13,B1] free
  x[A13,B2] free
  x[A13,B3] free
  x[A13,B4] free
  x[A13,B5] free
  x[A13,B6] free
  x[A13,B7] free
  x[A13,B8] free
  x[A14,B1] free
  x[A14,B2] free
  x[A14,B3] free
  x[A14,B4] free
  x[A14,B5] free
  x[A14,B6] free
  x[A14,B7] free
  x[A14,B8] free
  x[A15,B1] free
  x[A15,B2] free
  x[A15,B3] free
  x[A15,B4] free
  x[A15,B5] free
  x[A15,B6] free
  x[A15,B7] free
  x[A15,B8] free
Binaries
  y[A2] y[A3] y[A4] y[A5] y[A6] y[A7] y[A8] y[A9] y[A10] y[A11] y[A12] y[A13] y[A14] y[A15] y[A1]
End
