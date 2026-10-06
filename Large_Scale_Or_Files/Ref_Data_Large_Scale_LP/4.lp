\ Model ProductionOptimization
\ LP format - for model browsing. Use MPS format to capture full model detail.
\ Signature: 0x3fbdf11e0cd7c48
Maximize
  39.62 x[1] + 35.98 x[2] + 37.96 x[3] - 178539 y[1] - 157708 y[2]
   - 85192 y[3]
Subject To
 demand[1]: x[1] <= 5732
 demand[2]: x[2] <= 5607
 demand[3]: x[3] <= 4653
 quota[1]: x[1] <= 12892
 quota[2]: x[2] <= 7238
 quota[3]: x[3] <= 11902
 min_batch[1]: x[1] - 18 y[1] >= 0
 min_batch[2]: x[2] - 25 y[2] >= 0
 min_batch[3]: x[3] - 23 y[3] >= 0
 linking[1]: x[1] - 5732 y[1] <= 0
 linking[2]: x[2] - 5607 y[2] <= 0
 linking[3]: x[3] - 4653 y[3] <= 0
 shared_production_days: 0.00170648464163823 x[1]
   + 0.00303951367781155 x[2] + 0.00184842883548983 x[3] <= 22
Bounds
Binaries
 y[1] y[2] y[3]
Generals
 x[1] x[2] x[3]
End
