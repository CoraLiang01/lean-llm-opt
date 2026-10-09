\ Canonical native-bound formulation; bounds excluded from WL graph.
Minimize
  obj: + 20 x[A] + 24 x[B] + 16 x[C] + 28 x[D] + 1000 y[A] + 920 y[B] + 800 y[C] + 700 y[D]
Subject To
  Total_Demand: + 1 x[A] + 1 x[B] + 1 x[C] + 1 x[D] = 2000
  Linking[A]: + 1 x[A] - 900 y[A] <= 0
  Linking[B]: + 1 x[B] - 1000 y[B] <= 0
  Linking[C]: + 1 x[C] - 1200 y[C] <= 0
  Linking[D]: + 1 x[D] - 1600 y[D] <= 0
Bounds
  0 <= x[A] <= +inf
  0 <= x[B] <= +inf
  0 <= x[C] <= +inf
  0 <= x[D] <= +inf
  0 <= y[A] <= 1
  0 <= y[B] <= 1
  0 <= y[C] <= 1
  0 <= y[D] <= 1
Binaries
  y[A] y[B] y[C] y[D]
End
