\ Canonical explicit-bound formulation for WL comparison.
\ Finite bounds and singleton restrictions merged; original variable domains retained.
Minimize multi-objectives
  OBJ0: Priority=2 Weight=1 AbsTol=0 RelTol=0
   + 200 x[1] + 250 x[2] + 380 x[3] + 350 x[4] + 50 y[3] + 50 y[4]
  OBJ1: Priority=1 Weight=1 AbsTol=0 RelTol=0
   + 1 y[3] + 1 y[4]
Subject To
  Flour_Demand: + 30 x[1] + 20 x[2] + 40 x[3] + 10 x[4] = 100
  Rice_Demand: + 20 x[1] + 30 x[2] + 10 x[3] + 40 x[4] = 80
  Link_3: + 1 x[3] - 100 y[3] <= 0
  Link_4: + 1 x[4] - 100 y[4] <= 0
  __wl_bound_00001_lower: + 1 x[1] >= 0
  __wl_bound_00002_lower: + 1 x[2] >= 0
  __wl_bound_00003_lower: + 1 x[3] >= 0
  __wl_bound_00004_lower: + 1 x[4] >= 0
  __wl_bound_00005_lower: + 1 y[3] >= 0
  __wl_bound_00005_upper: + 1 y[3] <= 1
  __wl_bound_00006_lower: + 1 y[4] >= 0
  __wl_bound_00006_upper: + 1 y[4] <= 1
Bounds
  x[1] free
  x[2] free
  x[3] free
  x[4] free
Binaries
  y[3] y[4]
End
