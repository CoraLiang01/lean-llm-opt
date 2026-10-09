Let:
- \( x_{w,p} \): integer number of vehicles of product type \( p \) to be stored in warehouse \( w \) per day.

Indices:
- \( w \): Warehouse ID, in the order:
    1. Warehouse 1
    2. Warehouse 2
    3. Warehouse 3
    4. Warehouse 4
    5. Warehouse 5
    6. Warehouse 6
    7. Warehouse 7
    8. Warehouse 8
    9. Warehouse 9
    10. Warehouse 10
- \( p \): ProductName, in the order:
    1. Sedans
    2. SUVs
    3. Electric Vehicles
    4. Hybrid Vehicles
    5. Trucks
    6. Sports Cars
    7. Compact Cars
    8. Luxury Sedans
    9. Vans
    10. Pickup Trucks

Parameters:
- \( V_p \): Value per unit of product \( p \):

    | ProductName         | Value |
    |---------------------|-------|
    | Sedans              | 1200  |
    | SUVs                | 1800  |
    | Electric Vehicles   | 2500  |
    | Hybrid Vehicles     | 2000  |
    | Trucks              | 1500  |
    | Sports Cars         | 3000  |
    | Compact Cars        | 1000  |
    | Luxury Sedans       | 3500  |
    | Vans                | 1600  |
    | Pickup Trucks       | 1700  |

- \( W_p \): Weight per unit of product \( p \):

    | ProductName         | Weight |
    |---------------------|--------|
    | Sedans              | 20     |
    | SUVs                | 15     |
    | Electric Vehicles   | 25     |
    | Hybrid Vehicles     | 18     |
    | Trucks              | 10     |
    | Sports Cars         | 5      |
    | Compact Cars        | 22     |
    | Luxury Sedans       | 8      |
    | Vans                | 12     |
    | Pickup Trucks       | 7      |

- \( C_w \): Capacity of warehouse \( w \):

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

Model:

Decision variables:
- \( x_{w,p} \in \mathbb{Z}_{\geq 0} \) for all warehouses \( w \) and product types \( p \).

Objective:
\[
\max \sum_{w} \sum_{p} V_p \cdot x_{w,p}
\]

Subject to (for each warehouse \( w \)):
\[
\sum_{p} W_p \cdot x_{w,p} \leq C_w
\]
\[
x_{w,p} \in \mathbb{Z}_{\geq 0} \quad \forall w, p
\]

Where:
- \( x_{w,p} \): number of vehicles of product \( p \) to store in warehouse \( w \) per day (integer, nonnegative)
- \( V_p \): value per vehicle of product \( p \) (see table above)
- \( W_p \): weight per vehicle of product \( p \) (see table above)
- \( C_w \): capacity of warehouse \( w \) (see table above)

All indices and data are used in the original file order. No sorting or index resetting is performed. Each warehouse's capacity constraint depends only on its own \( x_{w,p} \) variables and the corresponding weights. The objective maximizes the total value of all vehicles stored across all warehouses. All decision variables are integer and nonnegative, as required.