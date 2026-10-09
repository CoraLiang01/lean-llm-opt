\ Canonical explicit-bound formulation for WL comparison.
\ Finite bounds and singleton restrictions merged; original variable domains retained.
Maximize
  obj: + 73 investment[1] + 76 investment[2] + 65 investment[3] + 70 investment[4] + 68 investment[5] + 75 investment[6] + 67 investment[7] + 74 investment[8] + 69 investment[9] + 72 investment[10]
Subject To
  Total_Investment: + 1 investment[1] + 1 investment[2] + 1 investment[3] + 1 investment[4] + 1 investment[5] + 1 investment[6] + 1 investment[7] + 1 investment[8] + 1 investment[9] + 1 investment[10] <= 35
  Risk_Market: + 5.0999999999999996 investment[1] + 6 investment[2] + 3.7999999999999998 investment[3] + 5.2999999999999998 investment[4] + 4.7999999999999998 investment[5] + 6.7000000000000002 investment[6] + 4 investment[7] + 5.7999999999999998 investment[8] + 4.7000000000000002 investment[9] + 5.5999999999999996 investment[10] <= 42
  Risk_Credit: + 6.2000000000000002 investment[1] + 7.0999999999999996 investment[2] + 5.7000000000000002 investment[3] + 6.4000000000000004 investment[4] + 6.0999999999999996 investment[5] + 6.5 investment[6] + 5.7999999999999998 investment[7] + 6.2999999999999998 investment[8] + 6 investment[9] + 6.5999999999999996 investment[10] <= 50
  Risk_Interest: + 3.3999999999999999 investment[1] + 3.7999999999999998 investment[2] + 2.8999999999999999 investment[3] + 3.2000000000000002 investment[4] + 3.1000000000000001 investment[5] + 4.0999999999999996 investment[6] + 3 investment[7] + 3.7000000000000002 investment[8] + 3.2999999999999998 investment[9] + 3.5 investment[10] <= 29
  Risk_FX: + 4.2000000000000002 investment[1] + 4.5999999999999996 investment[2] + 3.3999999999999999 investment[3] + 4.7999999999999998 investment[4] + 4 investment[5] + 5.0999999999999996 investment[6] + 3.7000000000000002 investment[7] + 4.9000000000000004 investment[8] + 4.0999999999999996 investment[9] + 4.5 investment[10] <= 39
  Risk_Liquidity: + 2.7999999999999998 investment[1] + 2.6000000000000001 investment[2] + 2.3999999999999999 investment[3] + 3.2999999999999998 investment[4] + 2.8999999999999999 investment[5] + 3.7000000000000002 investment[6] + 2.5 investment[7] + 3 investment[8] + 2.7999999999999998 investment[9] + 2.7000000000000002 investment[10] <= 25
  Risk_IR_Risk_r: + 5.9000000000000004 investment[1] + 4.7999999999999998 investment[2] + 6.2000000000000002 investment[3] + 5.0999999999999996 investment[4] + 4.9000000000000004 investment[5] + 6.2999999999999998 investment[6] + 5.5999999999999996 investment[7] + 6.0999999999999996 investment[8] + 5.2000000000000002 investment[9] + 5.4000000000000004 investment[10] <= 49
  Risk_Equity: + 2.7000000000000002 investment[1] + 2.5 investment[2] + 2.2000000000000002 investment[3] + 2.6000000000000001 investment[4] + 2.2999999999999998 investment[5] + 3 investment[6] + 2.3999999999999999 investment[7] + 2.8999999999999999 investment[8] + 2.7000000000000002 investment[9] + 2.7999999999999998 investment[10] <= 25
  Risk_Volatility: + 4.5 investment[1] + 5.0999999999999996 investment[2] + 3.8999999999999999 investment[3] + 4.7999999999999998 investment[4] + 4.4000000000000004 investment[5] + 5.4000000000000004 investment[6] + 4 investment[7] + 5 investment[8] + 4.2999999999999998 investment[9] + 4.7000000000000002 investment[10] <= 36
  Risk_Regulatory: + 2.2000000000000002 investment[1] + 3 investment[2] + 2.7000000000000002 investment[3] + 2.5 investment[4] + 2.3999999999999999 investment[5] + 3.2999999999999998 investment[6] + 2.6000000000000001 investment[7] + 3.2000000000000002 investment[8] + 2.2999999999999998 investment[9] + 2.7999999999999998 investment[10] <= 27
  __wl_bound_00001_lower: + 1 investment[1] >= 0
  __wl_bound_00001_upper: + 1 investment[1] <= 12
  __wl_bound_00002_lower: + 1 investment[2] >= 0
  __wl_bound_00002_upper: + 1 investment[2] <= 8
  __wl_bound_00003_lower: + 1 investment[3] >= 0
  __wl_bound_00003_upper: + 1 investment[3] <= 10
  __wl_bound_00004_lower: + 1 investment[4] >= 0
  __wl_bound_00004_upper: + 1 investment[4] <= 9
  __wl_bound_00005_lower: + 1 investment[5] >= 0
  __wl_bound_00005_upper: + 1 investment[5] <= 7
  __wl_bound_00006_lower: + 1 investment[6] >= 0
  __wl_bound_00006_upper: + 1 investment[6] <= 11
  __wl_bound_00007_lower: + 1 investment[7] >= 0
  __wl_bound_00007_upper: + 1 investment[7] <= 8
  __wl_bound_00008_lower: + 1 investment[8] >= 0
  __wl_bound_00008_upper: + 1 investment[8] <= 10
  __wl_bound_00009_lower: + 1 investment[9] >= 0
  __wl_bound_00009_upper: + 1 investment[9] <= 9
  __wl_bound_00010_lower: + 1 investment[10] >= 0
  __wl_bound_00010_upper: + 1 investment[10] <= 12
Bounds
  investment[1] free
  investment[2] free
  investment[3] free
  investment[4] free
  investment[5] free
  investment[6] free
  investment[7] free
  investment[8] free
  investment[9] free
  investment[10] free
End
