##### Decision Variables

Let $x_{ij} \in \mathbb{Z}_{\geq 0}$ denote the number of vehicles of type $i$ stored in warehouse $j$, where $i$ indexes vehicle types and $j$ indexes warehouses.

##### Parameters

- Vehicle types $i \in I$:
  - Sedans: Value $=1200$, Weight $=20$
  - SUVs: Value $=1800$, Weight $=15$
  - Electric Vehicles: Value $=2500$, Weight $=25$
  - Hybrid Vehicles: Value $=2000$, Weight $=18$
  - Trucks: Value $=1500$, Weight $=10$
  - Sports Cars: Value $=3000$, Weight $=5$
  - Compact Cars: Value $=1000$, Weight $=22$
  - Luxury Sedans: Value $=3500$, Weight $=8$
  - Vans: Value $=1600$, Weight $=12$
  - Pickup Trucks: Value $=1700$, Weight $=7$

- Warehouses $j \in J$:
  - Warehouse 1: Capacity $=100$
  - Warehouse 2: Capacity $=80$
  - Warehouse 3: Capacity $=120$
  - Warehouse 4: Capacity $=90$
  - Warehouse 5: Capacity $=50$
  - Warehouse 6: Capacity $=30$
  - Warehouse 7: Capacity $=110$
  - Warehouse 8: Capacity $=40$
  - Warehouse 9: Capacity $=60$
  - Warehouse 10: Capacity $=35$

Let $v_i$ be the value per unit of vehicle type $i$, $w_i$ be the weight per unit of vehicle type $i$, and $C_j$ be the capacity of warehouse $j$.

##### Objective Function

\[
\max \sum_{j \in J} \sum_{i \in I} v_i x_{ij}
\]

##### Constraints

1. Warehouse capacity constraints:
   \[
   \sum_{i \in I} w_i x_{ij} \leq C_j, \quad \forall j \in J
   \]

2. Integer and nonnegativity constraints:
   \[
   x_{ij} \in \mathbb{Z}_{\geq 0}, \quad \forall i \in I,\, j \in J
   \]

##### Sets and Parameters (explicit listing)

- $I = \{$Sedans, SUVs, Electric Vehicles, Hybrid Vehicles, Trucks, Sports Cars, Compact Cars, Luxury Sedans, Vans, Pickup Trucks$\}$
- $J = \{$Warehouse 1, Warehouse 2, Warehouse 3, Warehouse 4, Warehouse 5, Warehouse 6, Warehouse 7, Warehouse 8, Warehouse 9, Warehouse 10$\}$

- $v_i$ (Value per unit):
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

- $w_i$ (Weight per unit):
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

- $C_j$ (Warehouse capacities):
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