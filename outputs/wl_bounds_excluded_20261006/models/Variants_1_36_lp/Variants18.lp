\ Canonical native-bound formulation; bounds excluded from WL graph.
Minimize
  obj: + 1 y_P1 + 1 y_P2 + 1 y_P3 + 1 y_P4 + 1 y_P5 + 1 y_P6 + 1 y_P7 + 1 y_P8 + 1 y_P9 + 1 y_P10
Subject To
  demand_A: + 3 y_P1 + 1 y_P5 + 2 y_P6 + 1 y_P8 + 1 y_P9 >= 18
  demand_B: + 2 y_P2 + 1 y_P5 + 1 y_P7 + 2 y_P9 >= 14
  demand_C: + 1 y_P2 + 2 y_P3 + 1 y_P6 + 1 y_P7 + 1 y_P10 >= 12
  demand_D: + 1 y_P3 + 2 y_P4 + 1 y_P5 + 1 y_P8 + 1 y_P10 >= 10
  demand_E: + 1 y_P4 + 1 y_P7 + 1 y_P8 + 1 y_P10 >= 8
Bounds
  0 <= y_P1 <= +inf
  0 <= y_P2 <= +inf
  0 <= y_P3 <= +inf
  0 <= y_P4 <= +inf
  0 <= y_P5 <= +inf
  0 <= y_P6 <= +inf
  0 <= y_P7 <= +inf
  0 <= y_P8 <= +inf
  0 <= y_P9 <= +inf
  0 <= y_P10 <= +inf
Generals
  y_P1 y_P2 y_P3 y_P4 y_P5 y_P6 y_P7 y_P8 y_P9 y_P10
End
