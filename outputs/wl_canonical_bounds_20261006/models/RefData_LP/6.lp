\ Canonical explicit-bound formulation for WL comparison.
\ Finite bounds and singleton restrictions merged; original variable domains retained.
Maximize
  obj: - 5 x[A,1] + 35 x[A,2] + 25 x[A,3] + 5 x[B,1] + 45 x[B,2] + 35 x[B,3] - 5 x[C,1] + 35 x[C,2] + 25 x[C,3]
Subject To
  Material_Availability[1]: + 1 x[A,1] + 1 x[B,1] + 1 x[C,1] <= 100
  Material_Availability[2]: + 1 x[A,2] + 1 x[B,2] + 1 x[C,2] <= 100
  Material_Availability[3]: + 1 x[A,3] + 1 x[B,3] + 1 x[C,3] <= 60
  ProdA_Spec1: + 0.5 x[A,1] - 0.5 x[A,2] - 0.5 x[A,3] >= 0
  ProdA_Spec2: - 0.25 x[A,1] + 0.75 x[A,2] - 0.25 x[A,3] <= 0
  ProdB_Spec1: + 0.75 x[B,1] - 0.25 x[B,2] - 0.25 x[B,3] >= 0
  ProdB_Spec2: - 0.5 x[B,1] + 0.5 x[B,2] - 0.5 x[B,3] <= 0
  __wl_bound_00001_lower: + 1 x[A,1] >= 0
  __wl_bound_00002_lower: + 1 x[A,2] >= 0
  __wl_bound_00003_lower: + 1 x[A,3] >= 0
  __wl_bound_00004_lower: + 1 x[B,1] >= 0
  __wl_bound_00005_lower: + 1 x[B,2] >= 0
  __wl_bound_00006_lower: + 1 x[B,3] >= 0
  __wl_bound_00007_lower: + 1 x[C,1] >= 0
  __wl_bound_00008_lower: + 1 x[C,2] >= 0
  __wl_bound_00009_lower: + 1 x[C,3] >= 0
Bounds
  x[A,1] free
  x[A,2] free
  x[A,3] free
  x[B,1] free
  x[B,2] free
  x[B,3] free
  x[C,1] free
  x[C,2] free
  x[C,3] free
End
