\ Canonical native-bound formulation; bounds excluded from WL graph.
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
Bounds
  0 <= x[A,1] <= +inf
  0 <= x[A,2] <= +inf
  0 <= x[A,3] <= +inf
  0 <= x[B,1] <= +inf
  0 <= x[B,2] <= +inf
  0 <= x[B,3] <= +inf
  0 <= x[C,1] <= +inf
  0 <= x[C,2] <= +inf
  0 <= x[C,3] <= +inf
End
