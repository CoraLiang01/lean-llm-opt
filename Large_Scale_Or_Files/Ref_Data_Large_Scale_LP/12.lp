\ Model CashFlowPlanning
\ LP format - for model browsing. Use MPS format to capture full model detail.
\ Signature: 0xb5aedfcba7a03ec
Minimize
  28 x[1,1] + 45 x[1,2] + 60 x[1,3] + 73 x[1,4] + 28 x[2,1] + 45 x[2,2]
   + 60 x[2,3] + 28 x[3,1] + 45 x[3,2] + 28 x[4,1]
Subject To
 demand_month_1: x[1,1] + x[1,2] + x[1,3] + x[1,4] >= 1500
 demand_month_2: x[1,2] + x[1,3] + x[1,4] + x[2,1] + x[2,2] + x[2,3]
   >= 1000
 demand_month_3: x[1,3] + x[1,4] + x[2,2] + x[2,3] + x[3,1] + x[3,2]
   >= 2000
 demand_month_4: x[1,4] + x[2,3] + x[3,2] + x[4,1] >= 1200
Bounds
End
