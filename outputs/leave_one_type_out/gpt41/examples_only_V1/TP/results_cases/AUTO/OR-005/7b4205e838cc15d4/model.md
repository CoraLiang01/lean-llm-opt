Let x_{i,j} be the quantity shipped from supplier i to customer group j, where i ∈ {1,...,8} and j ∈ {1,...,8}.

Indices:
- Suppliers: supply1, supply2, supply3, supply4, supply5, supply6, supply7, supply8
- Customers: demand1, demand2, demand3, demand4, demand5, demand6, demand7, demand8

Parameters:
- demand_j: demand for customer group j
    - demand1: 9
    - demand2: 66
    - demand3: 56
    - demand4: 17
    - demand5: 43
    - demand6: 62
    - demand7: 10
    - demand8: 37
- supply_capacity_i: supply capacity of supplier i
    - supply1: 60
    - supply2: 22
    - supply3: 16
    - supply4: 14
    - supply5: 19
    - supply6: 70
    - supply7: 60
    - supply8: 39
- c_{i,j}: transportation cost per unit from supplier i to customer group j

|           | demand1      | demand2      | demand3      | demand4      | demand5      | demand6      | demand7      | demand8      |
|-----------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|
| supply1   | 0.0302073664 | 229.50723505 | 198.62356558 | 12.99505064  | 211.20732124 | 134.94429850 | 9.822206399  | 11.39407754  |
| supply2   | 232.34691308 | 3.625872644  | 0.2860543415 | 45.73127693  | 2.830479656  | 107.05891033 | 299.96317913 | 23.79935436  |
| supply3   | 11.06193833  | 0.204199533  | 0.278944728  | 45.72191272  | 59.54895566  | 5.097536740  | 300.00118415 | 23.71128271  |
| supply4   | 235.17948357 | 43.79466896  | 40.70984678  | 0.0777449662 | 4.237728183  | 131.70915517 | 296.55587568 | 29.81094002  |
| supply5   | 211.85808746 | 47.60180877  | 50.04007716  | 86.14548807  | 0.0619789792 | 5.334551530  | 270.06290424 | 3.853933134  |
| supply6   | 6.455066336  | 88.16323623  | 5.047091672  | 151.46120287 | 5.290760161  | 0.0460220534 | 9.936706602  | 103.75460989 |
| supply7   | 174.27229047 | 250.58223529 | 253.90413042 | 16.23546732  | 12.64314051  | 175.06728241 | 2.983839625  | 317.06551939 |
| supply8   | 207.87006254 | 1.517168472  | 24.02723929  | 27.13399928  | 73.20672469  | 125.72910360 | 15.46310325  | 0.2016498751 |

Variables:
x_{i,j} ≥ 0, ∀ i ∈ {supply1,...,supply8}, j ∈ {demand1,...,demand8}

Objective:
Minimize total transportation cost:
minimize
0.0302073664 x_{supply1,demand1} + 229.50723505 x_{supply1,demand2} + 198.62356558 x_{supply1,demand3} + 12.99505064 x_{supply1,demand4} + 211.20732124 x_{supply1,demand5} + 134.94429850 x_{supply1,demand6} + 9.822206399 x_{supply1,demand7} + 11.39407754 x_{supply1,demand8}
+ 232.34691308 x_{supply2,demand1} + 3.625872644 x_{supply2,demand2} + 0.2860543415 x_{supply2,demand3} + 45.73127693 x_{supply2,demand4} + 2.830479656 x_{supply2,demand5} + 107.05891033 x_{supply2,demand6} + 299.96317913 x_{supply2,demand7} + 23.79935436 x_{supply2,demand8}
+ 11.06193833 x_{supply3,demand1} + 0.204199533 x_{supply3,demand2} + 0.278944728 x_{supply3,demand3} + 45.72191272 x_{supply3,demand4} + 59.54895566 x_{supply3,demand5} + 5.097536740 x_{supply3,demand6} + 300.00118415 x_{supply3,demand7} + 23.71128271 x_{supply3,demand8}
+ 235.17948357 x_{supply4,demand1} + 43.79466896 x_{supply4,demand2} + 40.70984678 x_{supply4,demand3} + 0.0777449662 x_{supply4,demand4} + 4.237728183 x_{supply4,demand5} + 131.70915517 x_{supply4,demand6} + 296.55587568 x_{supply4,demand7} + 29.81094002 x_{supply4,demand8}
+ 211.85808746 x_{supply5,demand1} + 47.60180877 x_{supply5,demand2} + 50.04007716 x_{supply5,demand3} + 86.14548807 x_{supply5,demand4} + 0.0619789792 x_{supply5,demand5} + 5.334551530 x_{supply5,demand6} + 270.06290424 x_{supply5,demand7} + 3.853933134 x_{supply5,demand8}
+ 6.455066336 x_{supply6,demand1} + 88.16323623 x_{supply6,demand2} + 5.047091672 x_{supply6,demand3} + 151.46120287 x_{supply6,demand4} + 5.290760161 x_{supply6,demand5} + 0.0460220534 x_{supply6,demand6} + 9.936706602 x_{supply6,demand7} + 103.75460989 x_{supply6,demand8}
+ 174.27229047 x_{supply7,demand1} + 250.58223529 x_{supply7,demand2} + 253.90413042 x_{supply7,demand3} + 16.23546732 x_{supply7,demand4} + 12.64314051 x_{supply7,demand5} + 175.06728241 x_{supply7,demand6} + 2.983839625 x_{supply7,demand7} + 317.06551939 x_{supply7,demand8}
+ 207.87006254 x_{supply8,demand1} + 1.517168472 x_{supply8,demand2} + 24.02723929 x_{supply8,demand3} + 27.13399928 x_{supply8,demand4} + 73.20672469 x_{supply8,demand5} + 125.72910360 x_{supply8,demand6} + 15.46310325 x_{supply8,demand7} + 0.2016498751 x_{supply8,demand8}

Subject to:

1. Demand satisfaction for each customer group:
   For each j ∈ {demand1,...,demand8}:
   ∑_{i=supply1}^{supply8} x_{i,j} = demand_j

   - x_{supply1,demand1} + x_{supply2,demand1} + ... + x_{supply8,demand1} = 9
   - x_{supply1,demand2} + x_{supply2,demand2} + ... + x_{supply8,demand2} = 66
   - x_{supply1,demand3} + x_{supply2,demand3} + ... + x_{supply8,demand3} = 56
   - x_{supply1,demand4} + x_{supply2,demand4} + ... + x_{supply8,demand4} = 17
   - x_{supply1,demand5} + x_{supply2,demand5} + ... + x_{supply8,demand5} = 43
   - x_{supply1,demand6} + x_{supply2,demand6} + ... + x_{supply8,demand6} = 62
   - x_{supply1,demand7} + x_{supply2,demand7} + ... + x_{supply8,demand7} = 10
   - x_{supply1,demand8} + x_{supply2,demand8} + ... + x_{supply8,demand8} = 37

2. Supply capacity for each supplier:
   For each i ∈ {supply1,...,supply8}:
   ∑_{j=demand1}^{demand8} x_{i,j} ≤ supply_capacity_i

   - x_{supply1,demand1} + x_{supply1,demand2} + ... + x_{supply1,demand8} ≤ 60
   - x_{supply2,demand1} + x_{supply2,demand2} + ... + x_{supply2,demand8} ≤ 22
   - x_{supply3,demand1} + x_{supply3,demand2} + ... + x_{supply3,demand8} ≤ 16
   - x_{supply4,demand1} + x_{supply4,demand2} + ... + x_{supply4,demand8} ≤ 14
   - x_{supply5,demand1} + x_{supply5,demand2} + ... + x_{supply5,demand8} ≤ 19
   - x_{supply6,demand1} + x_{supply6,demand2} + ... + x_{supply6,demand8} ≤ 70
   - x_{supply7,demand1} + x_{supply7,demand2} + ... + x_{supply7,demand8} ≤ 60
   - x_{supply8,demand1} + x_{supply8,demand2} + ... + x_{supply8,demand8} ≤ 39

3. Non-negativity:
   x_{i,j} ≥ 0 for all i, j

This is a complete numerical linear programming formulation for the described transportation problem.