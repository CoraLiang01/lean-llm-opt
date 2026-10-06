Let:
- \( x_{w,p} \): Number of vehicles of product \( p \) to store in warehouse \( w \) per day (integer, \( x_{w,p} \geq 0 \))
- \( w \) indexes warehouses (from capacity.csv, Warehouse 1 to Warehouse 10)
- \( p \) indexes products (from products.csv: Sedans, SUVs, Electric Vehicles, Hybrid Vehicles, Trucks, Sports Cars, Compact Cars, Luxury Sedans, Vans, Pickup Trucks)

Parameters:
- \( V_p \): Value per unit of product \( p \) (from products.csv)
- \( W_p \): Weight per unit of product \( p \) (from products.csv)
- \( C_w \): Capacity of warehouse \( w \) (from capacity.csv)

Data:

From capacity.csv:
| Warehouse ID   | Capacity |
|----------------|----------|
| Warehouse 1    | 100      |
| Warehouse 2    | 80       |
| Warehouse 3    | 120      |
| Warehouse 4    | 90       |
| Warehouse 5    | 50       |
| Warehouse 6    | 30       |
| Warehouse 7    | 110      |
| Warehouse 8    | 40       |
| Warehouse 9    | 60       |
| Warehouse 10   | 35       |

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

Model:

Decision variables:
\[
x_{w,p} \in \mathbb{Z}_+, \quad \forall w \in \{\text{Warehouse 1}, \ldots, \text{Warehouse 10}\},\ p \in \{\text{Sedans}, \ldots, \text{Pickup Trucks}\}
\]

Objective:
\[
\max \sum_{w} \sum_{p} V_p \cdot x_{w,p}
\]
where \( V_p \) is the value for each product as above.

Subject to (for each warehouse \( w \)):
\[
\sum_{p} W_p \cdot x_{w,p} \leq C_w, \quad \forall w
\]
where \( W_p \) is the weight for each product and \( C_w \) is the capacity for each warehouse as above.

Variable domains:
\[
x_{w,p} \in \{0, 1, 2, \ldots\}
\]

Summary:
- The model maximizes the total value of cars stored across all warehouses.
- For each warehouse, the total weight of cars stored cannot exceed its capacity.
- The number of cars of each type stored in each warehouse per day must be a nonnegative integer.