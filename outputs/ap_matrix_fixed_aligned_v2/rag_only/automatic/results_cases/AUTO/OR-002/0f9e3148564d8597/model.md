Sets:
- Stores: S = {S1, S2, S3, S4, S5, S6, S7, S8, S9, S10, S11}
- Customers: C = {C1, C2, C3, C4, C5, C6, C7, C8, C9, C10, C11, C12}

Parameters (from source order):

Customer Demands (from customer_demand.csv):
- demand_C1 = 11
- demand_C2 = 1148
- demand_C3 = 54
- demand_C4 = 833
- demand_C5 = 154
- demand_C6 = 551
- demand_C7 = 7081
- demand_C8 = 76
- demand_C9 = 66
- demand_C10 = 174
- demand_C11 = 15
- demand_C12 = 680

Store Supply Capacities (from supply_capacity.csv):
- supply_S1 = 4
- supply_S2 = 575
- supply_S3 = 1504
- supply_S4 = 178
- supply_S5 = 228
- supply_S6 = 50
- supply_S7 = 3
- supply_S8 = 6148
- supply_S9 = 6
- supply_S10 = 10673
- supply_S11 = 174

Transportation Costs (from transportation_costs.csv, source order):

|         |   C1    |    C2    |    C3    |    C4    |    C5    |    C6    |    C7    |    C8    |    C9    |   C10    |   C11    |   C12    |
|---------|---------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|
| S1      | 0.6391  | 49.7184  | 33.7586  | 1570.6731| 1370.4095| 57.3531  | 57.1830  | 54.9210  |1143.6809 | 52.4913  |606.4434  |1192.4687 |
| S2      |605.4786 | 64.5356  |478.4779  | 887.0481 | 65.4611  | 71.9361  | 41.2902  | 70.3604  | 35.3589  |1472.7482 |  0.6005  | 49.8685  |
| S3      |1139.0440|  4.7851  |1805.6214 |1302.8958 |2437.3212 |103.8037  |774.6558  |  4.5160  |879.7049  |162.7056  |1208.6135 |110.1869  |
| S4      | 69.2699 |2105.4854 | 869.6820 |1494.8986 |310.5377  | 98.1546  |103.3692  |1758.8784 | 97.2854  | 94.6504  |1277.2515 | 21.6362  |
| S5      |980.4114 |899.3109  |1183.0326 | 402.0986 | 81.7886  |1115.6819 |123.8043  |1121.1469 |  0.0024  |1009.6452 | 35.3480  |1625.4346 |
| S6      |1246.7825|2105.7967 |1014.3393 |1494.6681 |362.0174  | 98.1714  |2170.4059 | 97.7319  | 97.2683  |1987.9908 | 70.9440  |389.1598  |
| S7      | 57.1086 | 23.8362  | 78.1057  | 742.8068 |1926.0797 |454.3790  |458.2901  |465.9308  | 28.1386  |524.6154  |997.5318  |104.4779  |
| S8      |981.2909 |120.9013  |1625.8207 |1267.8229 |2569.6446 | 13.4718  |815.1525  |253.4235  | 43.7656  |275.9784  |1228.0699 |103.4832  |
| S9      | 30.5328 |1444.8595 | 173.5547 |1307.3913 |965.2012  |1843.7769 |1483.6409 | 85.3221  |1353.5009 |1485.9154 | 29.4238  | 26.6194  |
| S10     | 94.1109 |1422.9971 |1470.7769 |1419.3382 | 38.9453  | 72.2011  |2040.4606 |1542.7026 |1803.8002 | 72.9437  |2181.4542 |973.5516  |
| S11     |1032.9074|166.3018  |1620.4767 | 64.6683  |2000.5092 |  0.0029  | 47.0384  | 52.9922  |1115.6336 |129.7934  |1295.0978 |2330.7682 |

Decision Variables:
Let x_{i,j} = quantity shipped from store i ∈ S to customer j ∈ C, for all i, j.
- Domain: x_{i,j} ≥ 0 (continuous, non-negative)

Objective:
Minimize total transportation cost:
minimize
∑_{i ∈ S} ∑_{j ∈ C} cost_{i,j} * x_{i,j}
where cost_{i,j} is as given in the table above.

Constraints:

1. Demand satisfaction (for each customer j ∈ C):
∑_{i ∈ S} x_{i,j} = demand_j

- For C1: ∑_{i ∈ S} x_{i,C1} = 11
- For C2: ∑_{i ∈ S} x_{i,C2} = 1148
- For C3: ∑_{i ∈ S} x_{i,C3} = 54
- For C4: ∑_{i ∈ S} x_{i,C4} = 833
- For C5: ∑_{i ∈ S} x_{i,C5} = 154
- For C6: ∑_{i ∈ S} x_{i,C6} = 551
- For C7: ∑_{i ∈ S} x_{i,C7} = 7081
- For C8: ∑_{i ∈ S} x_{i,C8} = 76
- For C9: ∑_{i ∈ S} x_{i,C9} = 66
- For C10: ∑_{i ∈ S} x_{i,C10} = 174
- For C11: ∑_{i ∈ S} x_{i,C11} = 15
- For C12: ∑_{i ∈ S} x_{i,C12} = 680

2. Supply capacity (for each store i ∈ S):
∑_{j ∈ C} x_{i,j} ≤ supply_i

- For S1: ∑_{j ∈ C} x_{S1,j} ≤ 4
- For S2: ∑_{j ∈ C} x_{S2,j} ≤ 575
- For S3: ∑_{j ∈ C} x_{S3,j} ≤ 1504
- For S4: ∑_{j ∈ C} x_{S4,j} ≤ 178
- For S5: ∑_{j ∈ C} x_{S5,j} ≤ 228
- For S6: ∑_{j ∈ C} x_{S6,j} ≤ 50
- For S7: ∑_{j ∈ C} x_{S7,j} ≤ 3
- For S8: ∑_{j ∈ C} x_{S8,j} ≤ 6148
- For S9: ∑_{j ∈ C} x_{S9,j} ≤ 6
- For S10: ∑_{j ∈ C} x_{S10,j} ≤ 10673
- For S11: ∑_{j ∈ C} x_{S11,j} ≤ 174

3. Non-negativity:
x_{i,j} ≥ 0 for all i ∈ S, j ∈ C

This is a complete numerical linear programming formulation of the transportation problem as described in the user query, using all identifiers and coefficients from the provided CSVs, preserving source order.