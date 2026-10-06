Let:
- I = {A1, A2, ..., A15} be the set of potential factory sites (indexed by i)
- J = {B1, B2, ..., B8} be the set of distribution centers (indexed by j)

Parameters:
- FixedCost_i: Fixed cost to open factory i (from facility_costs.csv)
- Capacity_i: Maximum capacity of factory i (from facility_costs.csv)
- Demand_j: Demand at distribution center j (from demand_requirements.csv)
- ShipCost_{i,j}: Variable shipping cost per unit from factory i to distribution center j (from shipping_costs.csv)

Decision Variables:
- y_i ∈ {0,1}: 1 if factory i is constructed, 0 otherwise
- x_{i,j} ≥ 0: Amount shipped from factory i to distribution center j

Objective:
Minimize total system cost (fixed + variable shipping):

Minimize:
Z = ∑_{i∈I} FixedCost_i * y_i + ∑_{i∈I} ∑_{j∈J} ShipCost_{i,j} * x_{i,j}

Subject to:
1. Demand satisfaction at each distribution center:
   ∑_{i∈I} x_{i,j} = Demand_j  ∀ j ∈ J

2. Factory capacity and open/close logic:
   ∑_{j∈J} x_{i,j} ≤ Capacity_i * y_i  ∀ i ∈ I

3. Binary and non-negativity:
   y_i ∈ {0,1} ∀ i ∈ I
   x_{i,j} ≥ 0 ∀ i ∈ I, j ∈ J

Parameter values:

Factories (I), with FixedCost and Capacity:
A1:  FixedCost = 0,   Capacity = 30
A2:  FixedCost = 175, Capacity = 10
A3:  FixedCost = 300, Capacity = 20
A4:  FixedCost = 375, Capacity = 30
A5:  FixedCost = 500, Capacity = 40
A6:  FixedCost = 200, Capacity = 20
A7:  FixedCost = 260, Capacity = 25
A8:  FixedCost = 220, Capacity = 30
A9:  FixedCost = 320, Capacity = 35
A10: FixedCost = 280, Capacity = 20
A11: FixedCost = 350, Capacity = 40
A12: FixedCost = 420, Capacity = 25
A13: FixedCost = 470, Capacity = 30
A14: FixedCost = 520, Capacity = 50
A15: FixedCost = 560, Capacity = 45

Distribution Centers (J), with Demand:
B1: 30
B2: 25
B3: 20
B4: 35
B5: 25
B6: 30
B7: 25
B8: 30

Shipping Cost Matrix ShipCost_{i,j} (rows: factories A1–A15, columns: B1–B8):

|      | B1 | B2 | B3 | B4 | B5 | B6 | B7 | B8 |
|------|----|----|----|----|----|----|----|----|
| A1   | 8  | 4  | 3  | 6  | 7  | 5  | 9  | 8  |
| A2   | 5  | 2  | 3  | 5  | 6  | 4  | 7  | 6  |
| A3   | 4  | 3  | 4  | 6  | 5  | 5  | 6  | 7  |
| A4   | 9  | 7  | 5  | 8  | 9  | 6  | 10 | 7  |
| A5   | 10 | 4  | 2  | 6  | 8  | 5  | 7  | 3  |
| A6   | 6  | 5  | 4  | 5  | 7  | 6  | 8  | 5  |
| A7   | 7  | 6  | 5  | 4  | 6  | 7  | 9  | 6  |
| A8   | 5  | 4  | 6  | 3  | 5  | 6  | 7  | 6  |
| A9   | 8  | 7  | 6  | 7  | 9  | 8  | 10 | 7  |
| A10  | 6  | 5  | 7  | 4  | 6  | 5  | 7  | 5  |
| A11  | 9  | 6  | 4  | 6  | 8  | 7  | 9  | 6  |
| A12  | 7  | 5  | 6  | 5  | 6  | 5  | 8  | 5  |
| A13  | 8  | 6  | 5  | 6  | 7  | 6  | 8  | 7  |
| A14  | 9  | 5  | 3  | 5  | 7  | 4  | 6  | 4  |
| A15  | 10 | 6  | 4  | 5  | 8  | 5  | 7  | 5  |

Summary:
The mathematical model for ElectroTech Manufacturing’s network restructuring is:

Minimize:
Z = ∑_{i∈I} FixedCost_i * y_i + ∑_{i∈I} ∑_{j∈J} ShipCost_{i,j} * x_{i,j}

Subject to:
- ∑_{i∈I} x_{i,j} = Demand_j  ∀ j ∈ J
- ∑_{j∈J} x_{i,j} ≤ Capacity_i * y_i  ∀ i ∈ I
- y_i ∈ {0,1} ∀ i ∈ I
- x_{i,j} ≥ 0 ∀ i ∈ I, j ∈ J

All parameter values (fixed costs, capacities, demands, shipping costs) are as listed above, directly from the CSV data.