Let $x_{ij}$ be the number of units of vehicle type $i$ to be stored in warehouse $j$. All $x_{ij}$ are nonnegative integers.

Indices:
- $i$ indexes vehicle types (as per ProductName in products.csv)
- $j$ indexes warehouses (as per Warehouse ID in capacity.csv)

Parameters:
- $p_i$: Value of vehicle type $i$ (from Value column)
- $w_i$: Weight (space requirement) of vehicle type $i$ (from Weight column)
- $C_j$: Capacity of warehouse $j$ (from Capacity column)

Vehicle types (in source order):
1. Sedans: $p_1 = 1200$, $w_1 = 20$
2. SUVs: $p_2 = 1800$, $w_2 = 15$
3. Electric Vehicles: $p_3 = 2500$, $w_3 = 25$
4. Hybrid Vehicles: $p_4 = 2000$, $w_4 = 18$
5. Trucks: $p_5 = 1500$, $w_5 = 10$
6. Sports Cars: $p_6 = 3000$, $w_6 = 5$
7. Compact Cars: $p_7 = 1000$, $w_7 = 22$
8. Luxury Sedans: $p_8 = 3500$, $w_8 = 8$
9. Vans: $p_9 = 1600$, $w_9 = 12$
10. Pickup Trucks: $p_{10} = 1700$, $w_{10} = 7$

Warehouses (in source order):
1. Warehouse 1: $C_1 = 100$
2. Warehouse 2: $C_2 = 80$
3. Warehouse 3: $C_3 = 120$
4. Warehouse 4: $C_4 = 90$
5. Warehouse 5: $C_5 = 50$
6. Warehouse 6: $C_6 = 30$
7. Warehouse 7: $C_7 = 110$
8. Warehouse 8: $C_8 = 40$
9. Warehouse 9: $C_9 = 60$
10. Warehouse 10: $C_{10} = 35$

Objective:
\[
\max \sum_{i=1}^{10} \sum_{j=1}^{10} p_i \cdot x_{ij}
\]
where $p_i$ is as above.

Constraints:

For each warehouse $j = 1, \ldots, 10$:
\[
\sum_{i=1}^{10} w_i \cdot x_{ij} \leq C_j
\]
where $w_i$ and $C_j$ are as above.

Variable domains:
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i = 1,\ldots,10;\ j = 1,\ldots,10
\]

Explicitly, with all coefficients:

Let $x_{ij}$ denote the number of units of vehicle type $i$ (see below) stored in warehouse $j$ (see below).

Vehicle types ($i$):
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

Warehouses ($j$):
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

Parameters:
- $p = [1200, 1800, 2500, 2000, 1500, 3000, 1000, 3500, 1600, 1700]$
- $w = [20, 15, 25, 18, 10, 5, 22, 8, 12, 7]$
- $C = [100, 80, 120, 90, 50, 30, 110, 40, 60, 35]$

Full model:

\[
\max \sum_{i=1}^{10} \sum_{j=1}^{10} p_i \cdot x_{ij}
\]

Subject to, for each $j = 1,\ldots,10$:
\[
20x_{1j} + 15x_{2j} + 25x_{3j} + 18x_{4j} + 10x_{5j} + 5x_{6j} + 22x_{7j} + 8x_{8j} + 12x_{9j} + 7x_{10j} \leq C_j
\]

\[
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i = 1,\ldots,10;\ j = 1,\ldots,10
\]