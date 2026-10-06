Let x_{w,p} be the number of vehicles of product type p stored in warehouse w per day. These are nonnegative integer variables.

Indices:
- w: Warehouse ID, as listed in capacity.csv (Warehouse 1, ..., Warehouse 10)
- p: ProductName, as listed in products.csv (Sedans, SUVs, Electric Vehicles, Hybrid Vehicles, Trucks, Sports Cars, Compact Cars, Luxury Sedans, Vans, Pickup Trucks)

Parameters:
- Value_p: Value per unit of product p (from products.csv)
- Weight_p: Weight per unit of product p (from products.csv)
- Capacity_w: Capacity of warehouse w (from capacity.csv)

Variables:
- x_{w,p} ∈ {0, 1, 2, ...} for all warehouses w and products p

Objective:
Maximize total value stored across all warehouses and products:
\[
\text{Maximize} \quad \sum_{w \in \text{Warehouses}} \sum_{p \in \text{Products}} \text{Value}_p \cdot x_{w,p}
\]
That is,
\[
\text{Maximize} \quad
1200 \sum_{w} x_{w,\text{Sedans}}
+ 1800 \sum_{w} x_{w,\text{SUVs}}
+ 2500 \sum_{w} x_{w,\text{Electric Vehicles}}
+ 2000 \sum_{w} x_{w,\text{Hybrid Vehicles}}
+ 1500 \sum_{w} x_{w,\text{Trucks}}
+ 3000 \sum_{w} x_{w,\text{Sports Cars}}
+ 1000 \sum_{w} x_{w,\text{Compact Cars}}
+ 3500 \sum_{w} x_{w,\text{Luxury Sedans}}
+ 1600 \sum_{w} x_{w,\text{Vans}}
+ 1700 \sum_{w} x_{w,\text{Pickup Trucks}}
\]

Subject to:

For each warehouse w, the total weight of vehicles stored cannot exceed its capacity:
\[
\sum_{p \in \text{Products}} \text{Weight}_p \cdot x_{w,p} \leq \text{Capacity}_w \quad \forall w
\]
That is, for each warehouse:

Warehouse 1:
\[
20 x_{Warehouse 1,\text{Sedans}}
+ 15 x_{Warehouse 1,\text{SUVs}}
+ 25 x_{Warehouse 1,\text{Electric Vehicles}}
+ 18 x_{Warehouse 1,\text{Hybrid Vehicles}}
+ 10 x_{Warehouse 1,\text{Trucks}}
+ 5 x_{Warehouse 1,\text{Sports Cars}}
+ 22 x_{Warehouse 1,\text{Compact Cars}}
+ 8 x_{Warehouse 1,\text{Luxury Sedans}}
+ 12 x_{Warehouse 1,\text{Vans}}
+ 7 x_{Warehouse 1,\text{Pickup Trucks}}
\leq 100
\]

Warehouse 2:
(same as above, ≤ 80)

Warehouse 3:
(same as above, ≤ 120)

Warehouse 4:
(same as above, ≤ 90)

Warehouse 5:
(same as above, ≤ 50)

Warehouse 6:
(same as above, ≤ 30)

Warehouse 7:
(same as above, ≤ 110)

Warehouse 8:
(same as above, ≤ 40)

Warehouse 9:
(same as above, ≤ 60)

Warehouse 10:
(same as above, ≤ 35)

Variable domains:
\[
x_{w,p} \in \mathbb{Z}_{\geq 0} \quad \forall w, p
\]

Summary:
- Decision variables: x_{w,p} = number of vehicles of product p stored in warehouse w per day (integer, ≥ 0)
- Objective: Maximize total value across all warehouses and products
- Constraints: For each warehouse, total weight of stored vehicles cannot exceed its capacity
- All coefficients and identifiers are taken directly from the provided CSVs.