\ Canonical explicit-bound formulation for WL comparison.
\ Finite bounds and singleton restrictions merged; original variable domains retained.
Minimize
  obj: + 937.34065047577019 x[S1,C1] + 86.930194110278521 x[S1,C2] + 49.380161478878499 x[S2,C1] + 1726.062101362784 x[S2,C2] + 105.1815083048055 y[S1] + 112.1842388512642 y[S2]
Subject To
  demand[C1]: + 1 x[S1,C1] + 1 x[S2,C1] = 7564
  demand[C2]: + 1 x[S1,C2] + 1 x[S2,C2] = 5246
  activation[S1]: + 1 x[S1,C1] + 1 x[S1,C2] - 12810 y[S1] <= 0
  activation[S2]: + 1 x[S2,C1] + 1 x[S2,C2] - 12810 y[S2] <= 0
  __wl_bound_00001_lower: + 1 x[S1,C1] >= 0
  __wl_bound_00002_lower: + 1 x[S1,C2] >= 0
  __wl_bound_00003_lower: + 1 x[S2,C1] >= 0
  __wl_bound_00004_lower: + 1 x[S2,C2] >= 0
  __wl_bound_00005_lower: + 1 y[S1] >= 0
  __wl_bound_00005_upper: + 1 y[S1] <= 1
  __wl_bound_00006_lower: + 1 y[S2] >= 0
  __wl_bound_00006_upper: + 1 y[S2] <= 1
Bounds
  x[S1,C1] free
  x[S1,C2] free
  x[S2,C1] free
  x[S2,C2] free
Binaries
  y[S1] y[S2]
End
