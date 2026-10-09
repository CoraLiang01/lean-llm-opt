Let $x_{ij}$ be the number of units of vehicle type $i$ to be stored in warehouse $j$. All $x_{ij}$ are nonnegative integers.

Let $I$ be the set of vehicle types (indexed by ProductName), and $J$ be the set of warehouses (indexed by Warehouse ID).

Let $v_i$ be the Value of vehicle type $i$, and $w_i$ be the Weight of vehicle type $i$ (space required per unit).
Let $C_j$ be the Capacity of warehouse $j$.

Sets and Parameters (in source order):

Vehicle types ($i$):
- Sedans: $v_{\text{Sedans}} = 1200$, $w_{\text{Sedans}} = 20$
- SUVs: $v_{\text{SUVs}} = 1800$, $w_{\text{SUVs}} = 15$
- Electric Vehicles: $v_{\text{Electric Vehicles}} = 2500$, $w_{\text{Electric Vehicles}} = 25$
- Hybrid Vehicles: $v_{\text{Hybrid Vehicles}} = 2000$, $w_{\text{Hybrid Vehicles}} = 18$
- Trucks: $v_{\text{Trucks}} = 1500$, $w_{\text{Trucks}} = 10$
- Sports Cars: $v_{\text{Sports Cars}} = 3000$, $w_{\text{Sports Cars}} = 5$
- Compact Cars: $v_{\text{Compact Cars}} = 1000$, $w_{\text{Compact Cars}} = 22$
- Luxury Sedans: $v_{\text{Luxury Sedans}} = 3500$, $w_{\text{Luxury Sedans}} = 8$
- Vans: $v_{\text{Vans}} = 1600$, $w_{\text{Vans}} = 12$
- Pickup Trucks: $v_{\text{Pickup Trucks}} = 1700$, $w_{\text{Pickup Trucks}} = 7$

Warehouses ($j$):
- Warehouse 1: $C_{\text{Warehouse 1}} = 100$
- Warehouse 2: $C_{\text{Warehouse 2}} = 80$
- Warehouse 3: $C_{\text{Warehouse 3}} = 120$
- Warehouse 4: $C_{\text{Warehouse 4}} = 90$
- Warehouse 5: $C_{\text{Warehouse 5}} = 50$
- Warehouse 6: $C_{\text{Warehouse 6}} = 30$
- Warehouse 7: $C_{\text{Warehouse 7}} = 110$
- Warehouse 8: $C_{\text{Warehouse 8}} = 40$
- Warehouse 9: $C_{\text{Warehouse 9}} = 60$
- Warehouse 10: $C_{\text{Warehouse 10}} = 35$

Model:

Objective:
\[
\max \sum_{i \in I} \sum_{j \in J} v_i \cdot x_{ij}
\]

Subject to, for each warehouse $j \in J$:
\[
\sum_{i \in I} w_i \cdot x_{ij} \leq C_j
\]

\[
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in I,\, j \in J
\]

Where:
- $x_{ij}$: Number of units of vehicle type $i$ to store in warehouse $j$ (integer, $\geq 0$)
- $v_i$: Value of vehicle type $i$ (see above)
- $w_i$: Weight (space requirement) of vehicle type $i$ (see above)
- $C_j$: Capacity of warehouse $j$ (see above)

All identifiers and coefficients are as retrieved and preserved in source order.