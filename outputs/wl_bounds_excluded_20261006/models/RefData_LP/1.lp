\ Canonical native-bound formulation; bounds excluded from WL graph.
Minimize
  obj: + 3000 x[MA,P1] + 3200 x[MA,P2] + 3100 x[MA,P3] + 2800 x[MB,P1] + 3300 x[MB,P2] + 2900 x[MB,P3] + 2900 x[MC,P1] + 3100 x[MC,P2] + 3000 x[MC,P3]
Subject To
  worker[MA]: + 1 x[MA,P1] + 1 x[MA,P2] + 1 x[MA,P3] = 1
  worker[MB]: + 1 x[MB,P1] + 1 x[MB,P2] + 1 x[MB,P3] = 1
  worker[MC]: + 1 x[MC,P1] + 1 x[MC,P2] + 1 x[MC,P3] = 1
  project[P1]: + 1 x[MA,P1] + 1 x[MB,P1] + 1 x[MC,P1] = 1
  project[P2]: + 1 x[MA,P2] + 1 x[MB,P2] + 1 x[MC,P2] = 1
  project[P3]: + 1 x[MA,P3] + 1 x[MB,P3] + 1 x[MC,P3] = 1
Bounds
  0 <= x[MA,P1] <= 1
  0 <= x[MA,P2] <= 1
  0 <= x[MA,P3] <= 1
  0 <= x[MB,P1] <= 1
  0 <= x[MB,P2] <= 1
  0 <= x[MB,P3] <= 1
  0 <= x[MC,P1] <= 1
  0 <= x[MC,P2] <= 1
  0 <= x[MC,P3] <= 1
Binaries
  x[MA,P1] x[MA,P2] x[MA,P3] x[MB,P1] x[MB,P2] x[MB,P3] x[MC,P1] x[MC,P2] x[MC,P3]
End
