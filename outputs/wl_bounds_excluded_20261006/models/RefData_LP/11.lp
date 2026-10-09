\ Canonical native-bound formulation; bounds excluded from WL graph.
Maximize
  obj: + 1 N
Subject To
  Hours_A: + 1 x[A,1] + 1 x[A,2] + 1 x[A,3] <= 100
  Hours_B: + 1 x[B,1] + 1 x[B,2] + 1 x[B,3] <= 150
  Hours_C: + 1 x[C,1] + 1 x[C,2] + 1 x[C,3] <= 80
  Hours_D: + 1 x[D,1] + 1 x[D,2] + 1 x[D,3] <= 200
  Requirement_1: - 1 N + 10 x[A,1] + 15 x[B,1] + 20 x[C,1] + 10 x[D,1] >= 0
  Requirement_2: - 1 N + 15 x[A,2] + 10 x[B,2] + 5 x[C,2] + 15 x[D,2] >= 0
  Requirement_3: - 1 N + 5 x[A,3] + 5 x[B,3] + 10 x[C,3] + 20 x[D,3] >= 0
Bounds
  0 <= N <= +inf
  0 <= x[A,1] <= +inf
  0 <= x[A,2] <= +inf
  0 <= x[A,3] <= +inf
  0 <= x[B,1] <= +inf
  0 <= x[B,2] <= +inf
  0 <= x[B,3] <= +inf
  0 <= x[C,1] <= +inf
  0 <= x[C,2] <= +inf
  0 <= x[C,3] <= +inf
  0 <= x[D,1] <= +inf
  0 <= x[D,2] <= +inf
  0 <= x[D,3] <= +inf
End
