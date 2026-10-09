\ Canonical native-bound formulation; bounds excluded from WL graph.
Minimize
  obj: + 0.63 x[S1,C1] + 49.710000000000001 x[S1,C2] + 33.75 x[S1,C3] + 1570.6700000000001 x[S1,C4] + 605.47000000000003 x[S2,C1] + 64.530000000000001 x[S2,C2] + 478.47000000000003 x[S2,C3] + 887.03999999999996 x[S2,C4] + 1139.04 x[S3,C1] + 4.7800000000000002 x[S3,C2] + 1805.6199999999999 x[S3,C3] + 1302.8900000000001 x[S3,C4]
Subject To
  demand[C1]: + 1 x[S1,C1] + 1 x[S2,C1] + 1 x[S3,C1] >= 11
  demand[C2]: + 1 x[S1,C2] + 1 x[S2,C2] + 1 x[S3,C2] >= 1148
  demand[C3]: + 1 x[S1,C3] + 1 x[S2,C3] + 1 x[S3,C3] >= 54
  demand[C4]: + 1 x[S1,C4] + 1 x[S2,C4] + 1 x[S3,C4] >= 833
  supply[S1]: + 1 x[S1,C1] + 1 x[S1,C2] + 1 x[S1,C3] + 1 x[S1,C4] <= 4
  supply[S2]: + 1 x[S2,C1] + 1 x[S2,C2] + 1 x[S2,C3] + 1 x[S2,C4] <= 575
  supply[S3]: + 1 x[S3,C1] + 1 x[S3,C2] + 1 x[S3,C3] + 1 x[S3,C4] <= 1504
Bounds
  0 <= x[S1,C1] <= +inf
  0 <= x[S1,C2] <= +inf
  0 <= x[S1,C3] <= +inf
  0 <= x[S1,C4] <= +inf
  0 <= x[S2,C1] <= +inf
  0 <= x[S2,C2] <= +inf
  0 <= x[S2,C3] <= +inf
  0 <= x[S2,C4] <= +inf
  0 <= x[S3,C1] <= +inf
  0 <= x[S3,C2] <= +inf
  0 <= x[S3,C3] <= +inf
  0 <= x[S3,C4] <= +inf
End
