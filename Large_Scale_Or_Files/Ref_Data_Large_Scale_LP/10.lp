\ Model FixedCostProduction
\ LP format - for model browsing. Use MPS format to capture full model detail.
\ Signature: 0x14c847781b751896
Minimize
  20 x[A] + 24 x[B] + 16 x[C] + 28 x[D] + 1000 y[A] + 920 y[B] + 800 y[C]
   + 700 y[D]
Subject To
 Total_Demand: x[A] + x[B] + x[C] + x[D] = 2000
 Linking[A]: x[A] - 900 y[A] <= 0
 Linking[B]: x[B] - 1000 y[B] <= 0
 Linking[C]: x[C] - 1200 y[C] <= 0
 Linking[D]: x[D] - 1600 y[D] <= 0
Bounds
Binaries
 y[A] y[B] y[C] y[D]
End
