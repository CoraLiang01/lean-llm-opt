Let $x_{ij}$ be the number of units of vehicle type $i$ to be stored in warehouse $j$. All $x_{ij}$ are nonnegative integers.

Let $i$ index vehicle types (in the order from products.csv):

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

Let $j$ index warehouses (in the order from capacity.csv):

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

Let $v_i$ be the value per unit of vehicle type $i$:

- Sedans: $1200$
- SUVs: $1800$
- Electric Vehicles: $2500$
- Hybrid Vehicles: $2000$
- Trucks: $1500$
- Sports Cars: $3000$
- Compact Cars: $1000$
- Luxury Sedans: $3500$
- Vans: $1600$
- Pickup Trucks: $1700$

Let $w_i$ be the weight per unit of vehicle type $i$:

- Sedans: $20$
- SUVs: $15$
- Electric Vehicles: $25$
- Hybrid Vehicles: $18$
- Trucks: $10$
- Sports Cars: $5$
- Compact Cars: $22$
- Luxury Sedans: $8$
- Vans: $12$
- Pickup Trucks: $7$

Let $C_j$ be the capacity of warehouse $j$:

- Warehouse 1: $100$
- Warehouse 2: $80$
- Warehouse 3: $120$
- Warehouse 4: $90$
- Warehouse 5: $50$
- Warehouse 6: $30$
- Warehouse 7: $110$
- Warehouse 8: $40$
- Warehouse 9: $60$
- Warehouse 10: $35$

The mathematical model is:

Objective:
\[
\max \sum_{i=1}^{10} \sum_{j=1}^{10} v_i \cdot x_{ij}
\]
where $v_i$ is as above.

Subject to, for each warehouse $j=1,\ldots,10$:
\[
\sum_{i=1}^{10} w_i \cdot x_{ij} \leq C_j
\]
where $w_i$ and $C_j$ are as above.

Variable domains:
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i=1,\ldots,10;\ j=1,\ldots,10
\]

All coefficients and identifiers are as retrieved and in original order.