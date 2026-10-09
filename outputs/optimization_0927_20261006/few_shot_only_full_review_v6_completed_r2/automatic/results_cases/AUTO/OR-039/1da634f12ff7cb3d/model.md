Let $x_{ij}$ be the number of units of vehicle type $i$ (ProductName) stored in warehouse $j$ (Warehouse ID). All $x_{ij} \in \mathbb{Z}_{\geq 0}$.

**Parameters:**

- $V_i$: Value of vehicle type $i$ (from "Value" in products.csv)
- $w_i$: Weight (space requirement) of vehicle type $i$ (from "Weight" in products.csv)
- $C_j$: Capacity of warehouse $j$ (from "Capacity" in capacity.csv)

**Sets:**

- $i \in$ {Sedans, SUVs, Electric Vehicles, Hybrid Vehicles, Trucks, Sports Cars, Compact Cars, Luxury Sedans, Vans, Pickup Trucks}
- $j \in$ {Warehouse 1, Warehouse 2, Warehouse 3, Warehouse 4, Warehouse 5, Warehouse 6, Warehouse 7, Warehouse 8, Warehouse 9, Warehouse 10}

**Model:**

Objective:
\[
\max \sum_{i} \sum_{j} V_i \cdot x_{ij}
\]

Subject to, for each warehouse $j$:
\[
\sum_{i} w_i \cdot x_{ij} \leq C_j \quad \forall j
\]

\[
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i, j
\]

**Numerical Data:**

Vehicle types and their values and weights:
- Sedans: $V_{\text{Sedans}} = 1200$, $w_{\text{Sedans}} = 20$
- SUVs: $V_{\text{SUVs}} = 1800$, $w_{\text{SUVs}} = 15$
- Electric Vehicles: $V_{\text{Electric Vehicles}} = 2500$, $w_{\text{Electric Vehicles}} = 25$
- Hybrid Vehicles: $V_{\text{Hybrid Vehicles}} = 2000$, $w_{\text{Hybrid Vehicles}} = 18$
- Trucks: $V_{\text{Trucks}} = 1500$, $w_{\text{Trucks}} = 10$
- Sports Cars: $V_{\text{Sports Cars}} = 3000$, $w_{\text{Sports Cars}} = 5$
- Compact Cars: $V_{\text{Compact Cars}} = 1000$, $w_{\text{Compact Cars}} = 22$
- Luxury Sedans: $V_{\text{Luxury Sedans}} = 3500$, $w_{\text{Luxury Sedans}} = 8$
- Vans: $V_{\text{Vans}} = 1600$, $w_{\text{Vans}} = 12$
- Pickup Trucks: $V_{\text{Pickup Trucks}} = 1700$, $w_{\text{Pickup Trucks}} = 7$

Warehouses and their capacities:
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

**Full Model:**

\[
\max \sum_{i \in \{\text{Sedans}, \text{SUVs}, \text{Electric Vehicles}, \text{Hybrid Vehicles}, \text{Trucks}, \text{Sports Cars}, \text{Compact Cars}, \text{Luxury Sedans}, \text{Vans}, \text{Pickup Trucks}\}} \sum_{j \in \{\text{Warehouse 1}, \ldots, \text{Warehouse 10}\}} V_i \cdot x_{ij}
\]

Subject to, for each warehouse $j$:
\[
\sum_{i} w_i \cdot x_{ij} \leq C_j \quad \forall j \in \{\text{Warehouse 1}, \ldots, \text{Warehouse 10}\}
\]

\[
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i, j
\]