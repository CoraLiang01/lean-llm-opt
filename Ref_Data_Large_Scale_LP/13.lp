\ Model SupplierSelection
\ LP format - for model browsing. Use MPS format to capture full model detail.
\ Signature: 0xc367897ad61433ba
Minimize multi-objectives
  OBJ0: Priority=2 Weight=1 AbsTol=0 RelTol=0
   200 x[1] + 250 x[2] + 380 x[3] + 350 x[4] + 50 y[3] + 50 y[4]
  OBJ1: Priority=1 Weight=1 AbsTol=0 RelTol=0
   y[3] + y[4]
Subject To
 Flour_Demand: 30 x[1] + 20 x[2] + 40 x[3] + 10 x[4] = 100
 Rice_Demand: 20 x[1] + 30 x[2] + 10 x[3] + 40 x[4] = 80
 Link_3: x[3] - 100 y[3] <= 0
 Link_4: x[4] - 100 y[4] <= 0
Bounds
Binaries
 y[3] y[4]
End
