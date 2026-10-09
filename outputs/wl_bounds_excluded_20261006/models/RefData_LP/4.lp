\ Canonical native-bound formulation; bounds excluded from WL graph.
Maximize
  obj: + 39.619999999999997 x[1] + 35.979999999999997 x[2] + 37.960000000000001 x[3] - 178539 y[1] - 157708 y[2] - 85192 y[3]
Subject To
  min_batch[1]: + 1 x[1] - 18 y[1] >= 0
  min_batch[2]: + 1 x[2] - 25 y[2] >= 0
  min_batch[3]: + 1 x[3] - 23 y[3] >= 0
  linking[1]: + 1 x[1] - 5732 y[1] <= 0
  linking[2]: + 1 x[2] - 5607 y[2] <= 0
  linking[3]: + 1 x[3] - 4653 y[3] <= 0
  shared_production_days: + 0.00170648464163823 x[1] + 0.0030395136778115501 x[2] + 0.0018484288354898299 x[3] <= 22
Bounds
  0 <= x[1] <= 5732
  0 <= x[2] <= 5607
  0 <= x[3] <= 4653
  0 <= y[1] <= 1
  0 <= y[2] <= 1
  0 <= y[3] <= 1
Generals
  x[1] x[2] x[3]
Binaries
  y[1] y[2] y[3]
End
