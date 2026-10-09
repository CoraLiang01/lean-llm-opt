\ Canonical explicit-bound formulation for WL comparison.
\ Finite bounds and singleton restrictions merged; original variable domains retained.
Minimize
  obj: + 3000 x[MA,P1] + 3200 x[MA,P2] + 3100 x[MA,P3] + 2800 x[MB,P1] + 3300 x[MB,P2] + 2900 x[MB,P3] + 2900 x[MC,P1] + 3100 x[MC,P2] + 3000 x[MC,P3]
Subject To
  worker[MA]: + 1 x[MA,P1] + 1 x[MA,P2] + 1 x[MA,P3] = 1
  worker[MB]: + 1 x[MB,P1] + 1 x[MB,P2] + 1 x[MB,P3] = 1
  worker[MC]: + 1 x[MC,P1] + 1 x[MC,P2] + 1 x[MC,P3] = 1
  project[P1]: + 1 x[MA,P1] + 1 x[MB,P1] + 1 x[MC,P1] = 1
  project[P2]: + 1 x[MA,P2] + 1 x[MB,P2] + 1 x[MC,P2] = 1
  project[P3]: + 1 x[MA,P3] + 1 x[MB,P3] + 1 x[MC,P3] = 1
  __wl_bound_00001_lower: + 1 x[MA,P1] >= 0
  __wl_bound_00001_upper: + 1 x[MA,P1] <= 1
  __wl_bound_00002_lower: + 1 x[MA,P2] >= 0
  __wl_bound_00002_upper: + 1 x[MA,P2] <= 1
  __wl_bound_00003_lower: + 1 x[MA,P3] >= 0
  __wl_bound_00003_upper: + 1 x[MA,P3] <= 1
  __wl_bound_00004_lower: + 1 x[MB,P1] >= 0
  __wl_bound_00004_upper: + 1 x[MB,P1] <= 1
  __wl_bound_00005_lower: + 1 x[MB,P2] >= 0
  __wl_bound_00005_upper: + 1 x[MB,P2] <= 1
  __wl_bound_00006_lower: + 1 x[MB,P3] >= 0
  __wl_bound_00006_upper: + 1 x[MB,P3] <= 1
  __wl_bound_00007_lower: + 1 x[MC,P1] >= 0
  __wl_bound_00007_upper: + 1 x[MC,P1] <= 1
  __wl_bound_00008_lower: + 1 x[MC,P2] >= 0
  __wl_bound_00008_upper: + 1 x[MC,P2] <= 1
  __wl_bound_00009_lower: + 1 x[MC,P3] >= 0
  __wl_bound_00009_upper: + 1 x[MC,P3] <= 1
Bounds
Binaries
  x[MA,P1] x[MA,P2] x[MA,P3] x[MB,P1] x[MB,P2] x[MB,P3] x[MC,P1] x[MC,P2] x[MC,P3]
End
