\ Canonical native-bound formulation; bounds excluded from WL graph.
Minimize
  obj: + 28 x[1,1] + 45 x[1,2] + 60 x[1,3] + 73 x[1,4] + 28 x[2,1] + 45 x[2,2] + 60 x[2,3] + 28 x[3,1] + 45 x[3,2] + 28 x[4,1]
Subject To
  demand_month_1: + 1 x[1,1] + 1 x[1,2] + 1 x[1,3] + 1 x[1,4] >= 1500
  demand_month_2: + 1 x[1,2] + 1 x[1,3] + 1 x[1,4] + 1 x[2,1] + 1 x[2,2] + 1 x[2,3] >= 1000
  demand_month_3: + 1 x[1,3] + 1 x[1,4] + 1 x[2,2] + 1 x[2,3] + 1 x[3,1] + 1 x[3,2] >= 2000
  demand_month_4: + 1 x[1,4] + 1 x[2,3] + 1 x[3,2] + 1 x[4,1] >= 1200
Bounds
  0 <= x[1,1] <= +inf
  0 <= x[1,2] <= +inf
  0 <= x[1,3] <= +inf
  0 <= x[1,4] <= +inf
  0 <= x[2,1] <= +inf
  0 <= x[2,2] <= +inf
  0 <= x[2,3] <= +inf
  0 <= x[3,1] <= +inf
  0 <= x[3,2] <= +inf
  0 <= x[4,1] <= +inf
End
