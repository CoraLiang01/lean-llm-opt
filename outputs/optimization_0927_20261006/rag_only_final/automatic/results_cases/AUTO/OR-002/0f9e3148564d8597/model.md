Let S = {S1, S2, S3, S4, S5, S6, S7, S8, S9, S10, S11} be the set of Walmart stores, and C = {C1, C2, C3, C4, C5, C6, C7, C8, C9, C10, C11, C12} be the set of customer groups.

Decision variables:
x_{i,j} = quantity transported from store i ∈ S to customer group j ∈ C, for all i, j.
Domain: x_{i,j} ≥ 0, continuous.

Parameters (from CSVs, source order preserved):

Customer Demands (customer_demand.csv):
C1: 11
C2: 1148
C3: 54
C4: 833
C5: 154
C6: 551
C7: 7081
C8: 76
C9: 66
C10: 174
C11: 15
C12: 680

Store Supply Capacities (supply_capacity.csv):
S1: 4
S2: 575
S3: 1504
S4: 178
S5: 228
S6: 50
S7: 3
S8: 6148
S9: 6
S10: 10673
S11: 174

Transportation Costs (transportation_costs.csv):

|        |   C1   |    C2    |    C3    |    C4    |    C5    |    C6    |    C7    |    C8    |    C9    |   C10   |   C11   |   C12   |
|--------|--------|----------|----------|----------|----------|----------|----------|----------|----------|---------|---------|---------|
| S1     | 0.6391 | 49.7184  | 33.7586  | 1570.6731| 1370.4095| 57.3531  | 57.1830  | 54.9210  | 1143.6809| 52.4913 | 606.4434|1192.4687|
| S2     |605.4786| 64.5356  |478.4779  | 887.0481 | 65.4611  | 71.9361  | 41.2902  | 70.3604  | 35.3589  |1472.7482|  0.6005 | 49.8685 |
| S3     |1139.0440| 4.7851  |1805.6214 |1302.8958 |2437.3212 |103.8037  |774.6558  | 4.5160   |879.7049  |162.7056 |1208.6135|110.1869 |
| S4     | 69.2699|2105.4854 | 869.6820 |1494.8986 |310.5377  | 98.1546  |103.3692  |1758.8784 | 97.2854  | 94.6504 |1277.2515| 21.6362 |
| S5     |980.4114|899.3109  |1183.0326 | 402.0986 | 81.7886  |1115.6819 |123.8043  |1121.1469 |  0.0024  |1009.6452| 35.3480 |1625.4346|
| S6     |1246.7825|2105.7967|1014.3393 |1494.6681 |362.0174  | 98.1714  |2170.4059 | 97.7319  | 97.2683  |1987.9908| 70.9440 |389.1598 |
| S7     | 57.1086| 23.8362  | 78.1057  | 742.8068 |1926.0797 |454.3790  |458.2901  |465.9308  | 28.1386  |524.6154 |997.5318 |104.4779 |
| S8     |981.2909|120.9013  |1625.8207 |1267.8229 |2569.6446 | 13.4718  |815.1525  |253.4235  | 43.7656  |275.9784 |1228.0699|103.4832 |
| S9     | 30.5328|1444.8595 | 173.5547 |1307.3913 |965.2012  |1843.7769 |1483.6409 | 85.3221  |1353.5009 |1485.9154| 29.4238 | 26.6194 |
| S10    | 94.1109|1422.9971 |1470.7769 |1419.3382 | 38.9453  | 72.2011  |2040.4606 |1542.7026 |1803.8002 | 72.9437 |2181.4542|973.5516 |
| S11    |1032.9074|166.3018 |1620.4767 | 64.6683  |2000.5092 |  0.0029  | 47.0384  | 52.9922  |1115.6336 |129.7934 |1295.0978|2330.7682|

Mathematical Model:

Variables:
x_{i,j} ≥ 0 for all i ∈ S, j ∈ C

Objective:
Minimize total transportation cost:
minimize
∑_{i ∈ S} ∑_{j ∈ C} c_{i,j} x_{i,j}
where c_{i,j} are the costs as given above.

Constraints:

1. Demand satisfaction (for each customer group j):
  ∑_{i ∈ S} x_{i,j} = d_j  for all j ∈ C
  where d_j is the demand for customer group j as listed above.

2. Supply capacity (for each store i):
  ∑_{j ∈ C} x_{i,j} ≤ s_i  for all i ∈ S
  where s_i is the supply capacity for store i as listed above.

3. Non-negativity:
  x_{i,j} ≥ 0  for all i ∈ S, j ∈ C

All identifiers, coefficients, and bounds are preserved in source order. No data is omitted or aggregated.

This is a complete numerical formulation of the transportation problem as described.