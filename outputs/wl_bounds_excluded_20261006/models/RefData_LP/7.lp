\ Canonical native-bound formulation; bounds excluded from WL graph.
Minimize
  obj: 0 OutputValue_Coal
Subject To
  Coal_Balance: + 1 OutputValue_Coal - 0.40000000000000002 OutputValue_Power - 0.59999999999999998 OutputValue_Steel = 0
  Power_Balance: - 0.59999999999999998 OutputValue_Coal + 0.90000000000000002 OutputValue_Power - 0.20000000000000001 OutputValue_Steel = 0
  Steel_Balance: - 0.40000000000000002 OutputValue_Coal - 0.5 OutputValue_Power + 0.80000000000000004 OutputValue_Steel = 0
Bounds
  0 <= OutputValue_Coal <= +inf
  0 <= OutputValue_Power <= +inf
  OutputValue_Steel = 10000
End
