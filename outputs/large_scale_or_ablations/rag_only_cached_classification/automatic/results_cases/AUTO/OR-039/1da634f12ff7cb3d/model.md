Let:
- W = {Warehouse 1, Warehouse 2, Warehouse 3, Warehouse 4, Warehouse 5, Warehouse 6, Warehouse 7, Warehouse 8, Warehouse 9, Warehouse 10}
- P = {Sedans, SUVs, Electric Vehicles, Hybrid Vehicles, Trucks, Sports Cars, Compact Cars, Luxury Sedans, Vans, Pickup Trucks}

Parameters:
- Value_p: value per unit of product p (from products.csv)
- Weight_p: weight per unit of product p (from products.csv)
- Capacity_w: capacity of warehouse w (from capacity.csv)

Decision variables:
- x_{w,p}: number of units of product p to store in warehouse w (integer, x_{w,p} ≥ 0)

Data:
From products.csv:
| ProductName         | Value | Weight |
|---------------------|-------|--------|
| Sedans              | 1200  | 20     |
| SUVs                | 1800  | 15     |
| Electric Vehicles   | 2500  | 25     |
| Hybrid Vehicles     | 2000  | 18     |
| Trucks              | 1500  | 10     |
| Sports Cars         | 3000  | 5      |
| Compact Cars        | 1000  | 22     |
| Luxury Sedans       | 3500  | 8      |
| Vans                | 1600  | 12     |
| Pickup Trucks       | 1700  | 7      |

From capacity.csv:
| Warehouse ID | Capacity |
|--------------|----------|
| Warehouse 1  | 100      |
| Warehouse 2  | 80       |
| Warehouse 3  | 120      |
| Warehouse 4  | 90       |
| Warehouse 5  | 50       |
| Warehouse 6  | 30       |
| Warehouse 7  | 110      |
| Warehouse 8  | 40       |
| Warehouse 9  | 60       |
| Warehouse 10 | 35       |

Mathematical Model:

Variables:
For each warehouse w ∈ W and product p ∈ P:
 x_{w,p} ∈ {0, 1, 2, ...}

Objective:
Maximize total value stored across all warehouses and products:
\[
\text{Maximize} \quad Z = \sum_{w \in W} \sum_{p \in P} \text{Value}_p \cdot x_{w,p}
\]
That is,
\[
\text{Maximize} \quad
\sum_{w \in W} \Big[
1200\,x_{w,\text{Sedans}} +
1800\,x_{w,\text{SUVs}} +
2500\,x_{w,\text{Electric Vehicles}} +
2000\,x_{w,\text{Hybrid Vehicles}} +
1500\,x_{w,\text{Trucks}} +
3000\,x_{w,\text{Sports Cars}} +
1000\,x_{w,\text{Compact Cars}} +
3500\,x_{w,\text{Luxury Sedans}} +
1600\,x_{w,\text{Vans}} +
1700\,x_{w,\text{Pickup Trucks}}
\Big]
\]

Subject to:

For each warehouse w ∈ W:
\[
\sum_{p \in P} \text{Weight}_p \cdot x_{w,p} \leq \text{Capacity}_w
\]
That is, for each warehouse:

Warehouse 1:
\[
20\,x_{1,\text{Sedans}} +
15\,x_{1,\text{SUVs}} +
25\,x_{1,\text{Electric Vehicles}} +
18\,x_{1,\text{Hybrid Vehicles}} +
10\,x_{1,\text{Trucks}} +
5\,x_{1,\text{Sports Cars}} +
22\,x_{1,\text{Compact Cars}} +
8\,x_{1,\text{Luxury Sedans}} +
12\,x_{1,\text{Vans}} +
7\,x_{1,\text{Pickup Trucks}}
\leq 100
\]

Warehouse 2:
(same left-hand side, ≤ 80)

Warehouse 3:
(same left-hand side, ≤ 120)

Warehouse 4:
(same left-hand side, ≤ 90)

Warehouse 5:
(same left-hand side, ≤ 50)

Warehouse 6:
(same left-hand side, ≤ 30)

Warehouse 7:
(same left-hand side, ≤ 110)

Warehouse 8:
(same left-hand side, ≤ 40)

Warehouse 9:
(same left-hand side, ≤ 60)

Warehouse 10:
(same left-hand side, ≤ 35)

Variable domains:
\[
x_{w,p} \in \mathbb{Z}_{\geq 0} \quad \forall w \in W,\, p \in P
\]

Summary:
- Decision: how many units of each vehicle type to store in each warehouse (integer, nonnegative)
- Objective: maximize total value
- Constraints: for each warehouse, total weight of stored vehicles cannot exceed its capacity

This is a complete integer programming formulation using all provided data and explicit warehouse and product identifiers.