\ Canonical native-bound formulation; bounds excluded from WL graph.
Maximize
  obj: + 80 x1 + 125 x2
Subject To
  Steel: + 8 x1 + 5 x2 <= 3500
  Iron: + 6 x1 + 4 x2 <= 1800
  MachineHours: + 4 x1 + 5 x2 <= 2800
Bounds
  0 <= x1 <= +inf
  0 <= x2 <= +inf
End
