\ Canonical explicit-bound formulation for WL comparison.
\ Finite bounds and singleton restrictions merged; original variable domains retained.
Minimize
  obj: + 20 x[A] + 24 x[B] + 16 x[C] + 28 x[D] + 1000 y[A] + 920 y[B] + 800 y[C] + 700 y[D]
Subject To
  Total_Demand: + 1 x[A] + 1 x[B] + 1 x[C] + 1 x[D] = 2000
  Linking[A]: + 1 x[A] - 900 y[A] <= 0
  Linking[B]: + 1 x[B] - 1000 y[B] <= 0
  Linking[C]: + 1 x[C] - 1200 y[C] <= 0
  Linking[D]: + 1 x[D] - 1600 y[D] <= 0
  __wl_bound_00001_lower: + 1 x[A] >= 0
  __wl_bound_00002_lower: + 1 x[B] >= 0
  __wl_bound_00003_lower: + 1 x[C] >= 0
  __wl_bound_00004_lower: + 1 x[D] >= 0
  __wl_bound_00005_lower: + 1 y[A] >= 0
  __wl_bound_00005_upper: + 1 y[A] <= 1
  __wl_bound_00006_lower: + 1 y[B] >= 0
  __wl_bound_00006_upper: + 1 y[B] <= 1
  __wl_bound_00007_lower: + 1 y[C] >= 0
  __wl_bound_00007_upper: + 1 y[C] <= 1
  __wl_bound_00008_lower: + 1 y[D] >= 0
  __wl_bound_00008_upper: + 1 y[D] <= 1
Bounds
  x[A] free
  x[B] free
  x[C] free
  x[D] free
Binaries
  y[A] y[B] y[C] y[D]
End
