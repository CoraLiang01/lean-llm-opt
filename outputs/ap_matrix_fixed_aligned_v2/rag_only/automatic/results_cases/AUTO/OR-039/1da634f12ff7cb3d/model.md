Let:
- \( x_{w,p} \) = number of vehicles of product type \( p \) to store in warehouse \( w \) per day (integer, \( x_{w,p} \geq 0 \))

Sets:
- Warehouses \( w \in \{\text{Warehouse 1}, \text{Warehouse 2}, \ldots, \text{Warehouse 10}\} \)
- Products \( p \in \{\text{Sedans}, \text{SUVs}, \text{Electric Vehicles}, \text{Hybrid Vehicles}, \text{Trucks}, \text{Sports Cars}, \text{Compact Cars}, \text{Luxury Sedans}, \text{Vans}, \text{Pickup Trucks}\} \)

Parameters (from products.csv and capacity.csv, in source order):

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

| Warehouse ID   | Capacity |
|---------------|----------|
| Warehouse 1   | 100      |
| Warehouse 2   | 80       |
| Warehouse 3   | 120      |
| Warehouse 4   | 90       |
| Warehouse 5   | 50       |
| Warehouse 6   | 30       |
| Warehouse 7   | 110      |
| Warehouse 8   | 40       |
| Warehouse 9   | 60       |
| Warehouse 10  | 35       |

Model:

Decision variables:
- For each warehouse \( w \) and product \( p \), \( x_{w,p} \in \mathbb{Z}_+ \) (nonnegative integers)

Objective:
\[
\max \sum_{w} \sum_{p} \text{Value}_p \cdot x_{w,p}
\]
where \(\text{Value}_p\) is as given above for each product.

Constraints:
For each warehouse \( w \) (in the order given):

\[
\sum_{p} \text{Weight}_p \cdot x_{w,p} \leq \text{Capacity}_w
\]

That is, for each warehouse:

- Warehouse 1:
  \[
  20x_{\text{Warehouse 1},\text{Sedans}} + 15x_{\text{Warehouse 1},\text{SUVs}} + 25x_{\text{Warehouse 1},\text{Electric Vehicles}} + 18x_{\text{Warehouse 1},\text{Hybrid Vehicles}} + 10x_{\text{Warehouse 1},\text{Trucks}} + 5x_{\text{Warehouse 1},\text{Sports Cars}} + 22x_{\text{Warehouse 1},\text{Compact Cars}} + 8x_{\text{Warehouse 1},\text{Luxury Sedans}} + 12x_{\text{Warehouse 1},\text{Vans}} + 7x_{\text{Warehouse 1},\text{Pickup Trucks}} \leq 100
  \]
- Warehouse 2:
  \[
  20x_{\text{Warehouse 2},\text{Sedans}} + 15x_{\text{Warehouse 2},\text{SUVs}} + 25x_{\text{Warehouse 2},\text{Electric Vehicles}} + 18x_{\text{Warehouse 2},\text{Hybrid Vehicles}} + 10x_{\text{Warehouse 2},\text{Trucks}} + 5x_{\text{Warehouse 2},\text{Sports Cars}} + 22x_{\text{Warehouse 2},\text{Compact Cars}} + 8x_{\text{Warehouse 2},\text{Luxury Sedans}} + 12x_{\text{Warehouse 2},\text{Vans}} + 7x_{\text{Warehouse 2},\text{Pickup Trucks}} \leq 80
  \]
- Warehouse 3:
  \[
  20x_{\text{Warehouse 3},\text{Sedans}} + 15x_{\text{Warehouse 3},\text{SUVs}} + 25x_{\text{Warehouse 3},\text{Electric Vehicles}} + 18x_{\text{Warehouse 3},\text{Hybrid Vehicles}} + 10x_{\text{Warehouse 3},\text{Trucks}} + 5x_{\text{Warehouse 3},\text{Sports Cars}} + 22x_{\text{Warehouse 3},\text{Compact Cars}} + 8x_{\text{Warehouse 3},\text{Luxury Sedans}} + 12x_{\text{Warehouse 3},\text{Vans}} + 7x_{\text{Warehouse 3},\text{Pickup Trucks}} \leq 120
  \]
- Warehouse 4:
  \[
  20x_{\text{Warehouse 4},\text{Sedans}} + 15x_{\text{Warehouse 4},\text{SUVs}} + 25x_{\text{Warehouse 4},\text{Electric Vehicles}} + 18x_{\text{Warehouse 4},\text{Hybrid Vehicles}} + 10x_{\text{Warehouse 4},\text{Trucks}} + 5x_{\text{Warehouse 4},\text{Sports Cars}} + 22x_{\text{Warehouse 4},\text{Compact Cars}} + 8x_{\text{Warehouse 4},\text{Luxury Sedans}} + 12x_{\text{Warehouse 4},\text{Vans}} + 7x_{\text{Warehouse 4},\text{Pickup Trucks}} \leq 90
  \]
- Warehouse 5:
  \[
  20x_{\text{Warehouse 5},\text{Sedans}} + 15x_{\text{Warehouse 5},\text{SUVs}} + 25x_{\text{Warehouse 5},\text{Electric Vehicles}} + 18x_{\text{Warehouse 5},\text{Hybrid Vehicles}} + 10x_{\text{Warehouse 5},\text{Trucks}} + 5x_{\text{Warehouse 5},\text{Sports Cars}} + 22x_{\text{Warehouse 5},\text{Compact Cars}} + 8x_{\text{Warehouse 5},\text{Luxury Sedans}} + 12x_{\text{Warehouse 5},\text{Vans}} + 7x_{\text{Warehouse 5},\text{Pickup Trucks}} \leq 50
  \]
- Warehouse 6:
  \[
  20x_{\text{Warehouse 6},\text{Sedans}} + 15x_{\text{Warehouse 6},\text{SUVs}} + 25x_{\text{Warehouse 6},\text{Electric Vehicles}} + 18x_{\text{Warehouse 6},\text{Hybrid Vehicles}} + 10x_{\text{Warehouse 6},\text{Trucks}} + 5x_{\text{Warehouse 6},\text{Sports Cars}} + 22x_{\text{Warehouse 6},\text{Compact Cars}} + 8x_{\text{Warehouse 6},\text{Luxury Sedans}} + 12x_{\text{Warehouse 6},\text{Vans}} + 7x_{\text{Warehouse 6},\text{Pickup Trucks}} \leq 30
  \]
- Warehouse 7:
  \[
  20x_{\text{Warehouse 7},\text{Sedans}} + 15x_{\text{Warehouse 7},\text{SUVs}} + 25x_{\text{Warehouse 7},\text{Electric Vehicles}} + 18x_{\text{Warehouse 7},\text{Hybrid Vehicles}} + 10x_{\text{Warehouse 7},\text{Trucks}} + 5x_{\text{Warehouse 7},\text{Sports Cars}} + 22x_{\text{Warehouse 7},\text{Compact Cars}} + 8x_{\text{Warehouse 7},\text{Luxury Sedans}} + 12x_{\text{Warehouse 7},\text{Vans}} + 7x_{\text{Warehouse 7},\text{Pickup Trucks}} \leq 110
  \]
- Warehouse 8:
  \[
  20x_{\text{Warehouse 8},\text{Sedans}} + 15x_{\text{Warehouse 8},\text{SUVs}} + 25x_{\text{Warehouse 8},\text{Electric Vehicles}} + 18x_{\text{Warehouse 8},\text{Hybrid Vehicles}} + 10x_{\text{Warehouse 8},\text{Trucks}} + 5x_{\text{Warehouse 8},\text{Sports Cars}} + 22x_{\text{Warehouse 8},\text{Compact Cars}} + 8x_{\text{Warehouse 8},\text{Luxury Sedans}} + 12x_{\text{Warehouse 8},\text{Vans}} + 7x_{\text{Warehouse 8},\text{Pickup Trucks}} \leq 40
  \]
- Warehouse 9:
  \[
  20x_{\text{Warehouse 9},\text{Sedans}} + 15x_{\text{Warehouse 9},\text{SUVs}} + 25x_{\text{Warehouse 9},\text{Electric Vehicles}} + 18x_{\text{Warehouse 9},\text{Hybrid Vehicles}} + 10x_{\text{Warehouse 9},\text{Trucks}} + 5x_{\text{Warehouse 9},\text{Sports Cars}} + 22x_{\text{Warehouse 9},\text{Compact Cars}} + 8x_{\text{Warehouse 9},\text{Luxury Sedans}} + 12x_{\text{Warehouse 9},\text{Vans}} + 7x_{\text{Warehouse 9},\text{Pickup Trucks}} \leq 60
  \]
- Warehouse 10:
  \[
  20x_{\text{Warehouse 10},\text{Sedans}} + 15x_{\text{Warehouse 10},\text{SUVs}} + 25x_{\text{Warehouse 10},\text{Electric Vehicles}} + 18x_{\text{Warehouse 10},\text{Hybrid Vehicles}} + 10x_{\text{Warehouse 10},\text{Trucks}} + 5x_{\text{Warehouse 10},\text{Sports Cars}} + 22x_{\text{Warehouse 10},\text{Compact Cars}} + 8x_{\text{Warehouse 10},\text{Luxury Sedans}} + 12x_{\text{Warehouse 10},\text{Vans}} + 7x_{\text{Warehouse 10},\text{Pickup Trucks}} \leq 35
  \]

Variable domains:
\[
x_{w,p} \in \mathbb{Z}_+, \quad \forall w, p
\]

Summary:
- Maximize total value of cars stored across all warehouses.
- For each warehouse, the total weight of cars stored cannot exceed its capacity.
- All decision variables are nonnegative integers, indexed by both warehouse and product type, using explicit warehouse and product names as IDs. All coefficients and constraints are as given in the supplied data.