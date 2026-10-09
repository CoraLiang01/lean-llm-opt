Let:
- W = set of warehouses, indexed by w (from the "Warehouse ID" column in capacity.csv)
- P = set of products, indexed by p (from the "ProductName" column in products.csv)
- cap_w = capacity of warehouse w (from "Capacity" in capacity.csv)
- val_p = value per unit of product p (from "Value" in products.csv)
- wt_p = weight (space consumption) per unit of product p (from "Weight" in products.csv)
- x_{w,p} = number of units of product p to store in warehouse w (decision variable, integer, x_{w,p} ≥ 0)

Data (in supplied order):

Warehouses (from capacity.csv):
1. Warehouse 1, Capacity = 100
2. Warehouse 2, Capacity = 80
3. Warehouse 3, Capacity = 120
4. Warehouse 4, Capacity = 90
5. Warehouse 5, Capacity = 50
6. Warehouse 6, Capacity = 30
7. Warehouse 7, Capacity = 110
8. Warehouse 8, Capacity = 40
9. Warehouse 9, Capacity = 60
10. Warehouse 10, Capacity = 35

Products (from products.csv):
1. Sedans: Value = 1200, Weight = 20
2. SUVs: Value = 1800, Weight = 15
3. Electric Vehicles: Value = 2500, Weight = 25
4. Hybrid Vehicles: Value = 2000, Weight = 18
5. Trucks: Value = 1500, Weight = 10
6. Sports Cars: Value = 3000, Weight = 5
7. Compact Cars: Value = 1000, Weight = 22
8. Luxury Sedans: Value = 3500, Weight = 8
9. Vans: Value = 1600, Weight = 12
10. Pickup Trucks: Value = 1700, Weight = 7

Decision variables:
For each warehouse w ∈ {Warehouse 1, ..., Warehouse 10} and each product p ∈ {Sedans, SUVs, Electric Vehicles, Hybrid Vehicles, Trucks, Sports Cars, Compact Cars, Luxury Sedans, Vans, Pickup Trucks}:
 x_{w,p} ∈ {0, 1, 2, ...}

Objective:
Maximize total value:
\[
\text{Maximize} \quad \sum_{w \in W} \sum_{p \in P} \text{val}_p \cdot x_{w,p}
\]
That is,
\[
\text{Maximize} \quad \sum_{w} \Big[ 1200\,x_{w,\text{Sedans}} + 1800\,x_{w,\text{SUVs}} + 2500\,x_{w,\text{Electric Vehicles}} + 2000\,x_{w,\text{Hybrid Vehicles}} + 1500\,x_{w,\text{Trucks}} + 3000\,x_{w,\text{Sports Cars}} + 1000\,x_{w,\text{Compact Cars}} + 3500\,x_{w,\text{Luxury Sedans}} + 1600\,x_{w,\text{Vans}} + 1700\,x_{w,\text{Pickup Trucks}} \Big]
\]

Subject to (for each warehouse w):

Warehouse 1:
\[
20\,x_{\text{Warehouse 1},\text{Sedans}} + 15\,x_{\text{Warehouse 1},\text{SUVs}} + 25\,x_{\text{Warehouse 1},\text{Electric Vehicles}} + 18\,x_{\text{Warehouse 1},\text{Hybrid Vehicles}} + 10\,x_{\text{Warehouse 1},\text{Trucks}} + 5\,x_{\text{Warehouse 1},\text{Sports Cars}} + 22\,x_{\text{Warehouse 1},\text{Compact Cars}} + 8\,x_{\text{Warehouse 1},\text{Luxury Sedans}} + 12\,x_{\text{Warehouse 1},\text{Vans}} + 7\,x_{\text{Warehouse 1},\text{Pickup Trucks}} \leq 100
\]

Warehouse 2:
\[
20\,x_{\text{Warehouse 2},\text{Sedans}} + 15\,x_{\text{Warehouse 2},\text{SUVs}} + 25\,x_{\text{Warehouse 2},\text{Electric Vehicles}} + 18\,x_{\text{Warehouse 2},\text{Hybrid Vehicles}} + 10\,x_{\text{Warehouse 2},\text{Trucks}} + 5\,x_{\text{Warehouse 2},\text{Sports Cars}} + 22\,x_{\text{Warehouse 2},\text{Compact Cars}} + 8\,x_{\text{Warehouse 2},\text{Luxury Sedans}} + 12\,x_{\text{Warehouse 2},\text{Vans}} + 7\,x_{\text{Warehouse 2},\text{Pickup Trucks}} \leq 80
\]

Warehouse 3:
\[
20\,x_{\text{Warehouse 3},\text{Sedans}} + 15\,x_{\text{Warehouse 3},\text{SUVs}} + 25\,x_{\text{Warehouse 3},\text{Electric Vehicles}} + 18\,x_{\text{Warehouse 3},\text{Hybrid Vehicles}} + 10\,x_{\text{Warehouse 3},\text{Trucks}} + 5\,x_{\text{Warehouse 3},\text{Sports Cars}} + 22\,x_{\text{Warehouse 3},\text{Compact Cars}} + 8\,x_{\text{Warehouse 3},\text{Luxury Sedans}} + 12\,x_{\text{Warehouse 3},\text{Vans}} + 7\,x_{\text{Warehouse 3},\text{Pickup Trucks}} \leq 120
\]

Warehouse 4:
\[
20\,x_{\text{Warehouse 4},\text{Sedans}} + 15\,x_{\text{Warehouse 4},\text{SUVs}} + 25\,x_{\text{Warehouse 4},\text{Electric Vehicles}} + 18\,x_{\text{Warehouse 4},\text{Hybrid Vehicles}} + 10\,x_{\text{Warehouse 4},\text{Trucks}} + 5\,x_{\text{Warehouse 4},\text{Sports Cars}} + 22\,x_{\text{Warehouse 4},\text{Compact Cars}} + 8\,x_{\text{Warehouse 4},\text{Luxury Sedans}} + 12\,x_{\text{Warehouse 4},\text{Vans}} + 7\,x_{\text{Warehouse 4},\text{Pickup Trucks}} \leq 90
\]

Warehouse 5:
\[
20\,x_{\text{Warehouse 5},\text{Sedans}} + 15\,x_{\text{Warehouse 5},\text{SUVs}} + 25\,x_{\text{Warehouse 5},\text{Electric Vehicles}} + 18\,x_{\text{Warehouse 5},\text{Hybrid Vehicles}} + 10\,x_{\text{Warehouse 5},\text{Trucks}} + 5\,x_{\text{Warehouse 5},\text{Sports Cars}} + 22\,x_{\text{Warehouse 5},\text{Compact Cars}} + 8\,x_{\text{Warehouse 5},\text{Luxury Sedans}} + 12\,x_{\text{Warehouse 5},\text{Vans}} + 7\,x_{\text{Warehouse 5},\text{Pickup Trucks}} \leq 50
\]

Warehouse 6:
\[
20\,x_{\text{Warehouse 6},\text{Sedans}} + 15\,x_{\text{Warehouse 6},\text{SUVs}} + 25\,x_{\text{Warehouse 6},\text{Electric Vehicles}} + 18\,x_{\text{Warehouse 6},\text{Hybrid Vehicles}} + 10\,x_{\text{Warehouse 6},\text{Trucks}} + 5\,x_{\text{Warehouse 6},\text{Sports Cars}} + 22\,x_{\text{Warehouse 6},\text{Compact Cars}} + 8\,x_{\text{Warehouse 6},\text{Luxury Sedans}} + 12\,x_{\text{Warehouse 6},\text{Vans}} + 7\,x_{\text{Warehouse 6},\text{Pickup Trucks}} \leq 30
\]

Warehouse 7:
\[
20\,x_{\text{Warehouse 7},\text{Sedans}} + 15\,x_{\text{Warehouse 7},\text{SUVs}} + 25\,x_{\text{Warehouse 7},\text{Electric Vehicles}} + 18\,x_{\text{Warehouse 7},\text{Hybrid Vehicles}} + 10\,x_{\text{Warehouse 7},\text{Trucks}} + 5\,x_{\text{Warehouse 7},\text{Sports Cars}} + 22\,x_{\text{Warehouse 7},\text{Compact Cars}} + 8\,x_{\text{Warehouse 7},\text{Luxury Sedans}} + 12\,x_{\text{Warehouse 7},\text{Vans}} + 7\,x_{\text{Warehouse 7},\text{Pickup Trucks}} \leq 110
\]

Warehouse 8:
\[
20\,x_{\text{Warehouse 8},\text{Sedans}} + 15\,x_{\text{Warehouse 8},\text{SUVs}} + 25\,x_{\text{Warehouse 8},\text{Electric Vehicles}} + 18\,x_{\text{Warehouse 8},\text{Hybrid Vehicles}} + 10\,x_{\text{Warehouse 8},\text{Trucks}} + 5\,x_{\text{Warehouse 8},\text{Sports Cars}} + 22\,x_{\text{Warehouse 8},\text{Compact Cars}} + 8\,x_{\text{Warehouse 8},\text{Luxury Sedans}} + 12\,x_{\text{Warehouse 8},\text{Vans}} + 7\,x_{\text{Warehouse 8},\text{Pickup Trucks}} \leq 40
\]

Warehouse 9:
\[
20\,x_{\text{Warehouse 9},\text{Sedans}} + 15\,x_{\text{Warehouse 9},\text{SUVs}} + 25\,x_{\text{Warehouse 9},\text{Electric Vehicles}} + 18\,x_{\text{Warehouse 9},\text{Hybrid Vehicles}} + 10\,x_{\text{Warehouse 9},\text{Trucks}} + 5\,x_{\text{Warehouse 9},\text{Sports Cars}} + 22\,x_{\text{Warehouse 9},\text{Compact Cars}} + 8\,x_{\text{Warehouse 9},\text{Luxury Sedans}} + 12\,x_{\text{Warehouse 9},\text{Vans}} + 7\,x_{\text{Warehouse 9},\text{Pickup Trucks}} \leq 60
\]

Warehouse 10:
\[
20\,x_{\text{Warehouse 10},\text{Sedans}} + 15\,x_{\text{Warehouse 10},\text{SUVs}} + 25\,x_{\text{Warehouse 10},\text{Electric Vehicles}} + 18\,x_{\text{Warehouse 10},\text{Hybrid Vehicles}} + 10\,x_{\text{Warehouse 10},\text{Trucks}} + 5\,x_{\text{Warehouse 10},\text{Sports Cars}} + 22\,x_{\text{Warehouse 10},\text{Compact Cars}} + 8\,x_{\text{Warehouse 10},\text{Luxury Sedans}} + 12\,x_{\text{Warehouse 10},\text{Vans}} + 7\,x_{\text{Warehouse 10},\text{Pickup Trucks}} \leq 35
\]

Variable domains:
\[
x_{w,p} \in \mathbb{Z}_{\geq 0} \quad \forall w \in W,\, p \in P
\]

This is a complete integer programming formulation for the inventory-replenishment strategy, using all supplied data and respecting all requirements.