\ Canonical native-bound formulation; bounds excluded from WL graph.
Minimize
  obj: + 1 y_P1 + 1 y_P2 + 1 y_P3 + 1 y_P4 + 1 y_P5 + 1 y_P6 + 1 y_P7 + 1 y_P8 + 1 y_P9
Subject To
  demand_A: + 4 y_P1 + 2 y_P5 + 1 y_P6 + 1 y_P8 + 2 y_P9 >= 24
  demand_B: + 3 y_P2 + 1 y_P5 + 1 y_P7 + 1 y_P8 >= 18
  demand_C: + 2 y_P3 + 1 y_P6 + 1 y_P8 >= 12
  demand_D: + 2 y_P4 + 1 y_P7 + 1 y_P9 >= 10
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
Generals
  y_P1 y_P2 y_P3 y_P4 y_P5 y_P6 y_P7 y_P8 y_P9
End
