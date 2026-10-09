\ Canonical explicit-bound formulation for WL comparison.
\ Finite bounds and singleton restrictions merged; original variable domains retained.
Minimize
  obj: 0 OutputValue_Coal
Subject To
  Coal_Balance: + 1 OutputValue_Coal - 0.40000000000000002 OutputValue_Power - 0.59999999999999998 OutputValue_Steel = 0
  Power_Balance: - 0.59999999999999998 OutputValue_Coal + 0.90000000000000002 OutputValue_Power - 0.20000000000000001 OutputValue_Steel = 0
  Steel_Balance: - 0.40000000000000002 OutputValue_Coal - 0.5 OutputValue_Power + 0.80000000000000004 OutputValue_Steel = 0
  __wl_bound_00001_lower: + 1 OutputValue_Coal >= 0
  __wl_bound_00002_lower: + 1 OutputValue_Power >= 0
  __wl_bound_00003_lower: + 1 OutputValue_Steel >= 10000
  __wl_bound_00003_upper: + 1 OutputValue_Steel <= 10000
Bounds
  OutputValue_Coal free
  OutputValue_Power free
  OutputValue_Steel free
End
