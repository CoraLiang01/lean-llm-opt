##### Sets and Indices

- Let $P$ be the set of vehicle types (products):  
  $P = \{\text{Sedans},\ \text{SUVs},\ \text{Electric Vehicles},\ \text{Hybrid Vehicles},\ \text{Trucks},\ \text{Sports Cars},\ \text{Compact Cars},\ \text{Luxury Sedans},\ \text{Vans},\ \text{Pickup Trucks}\}$

- Let $W$ be the set of warehouses:  
  $W = \{\text{Warehouse 1},\ \text{Warehouse 2},\ \text{Warehouse 3},\ \text{Warehouse 4},\ \text{Warehouse 5},\ \text{Warehouse 6},\ \text{Warehouse 7},\ \text{Warehouse 8},\ \text{Warehouse 9},\ \text{Warehouse 10}\}$

##### Parameters

- Value per unit for each product $i \in P$:
  - $\text{Value}_i$:
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

- Weight per unit for each product $i \in P$:
  - $\text{Weight}_i$:
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

- Capacity for each warehouse $w \in W$:
  - $\text{Capacity}_w$:
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

##### Decision Variables

- $x_{iw} \in \mathbb{Z}_{\geq 0}$: Number of vehicles of type $i \in P$ to be stored in warehouse $w \in W$ (integer, nonnegative).

##### Objective Function

\[
\max \sum_{w \in W} \sum_{i \in P} \text{Value}_i \cdot x_{iw}
\]

##### Constraints

1. **Warehouse Capacity Constraints:**  
   For each warehouse $w \in W$,
   \[
   \sum_{i \in P} \text{Weight}_i \cdot x_{iw} \leq \text{Capacity}_w
   \]

2. **Nonnegativity and Integrality:**  
   For all $i \in P$, $w \in W$,
   \[
   x_{iw} \in \mathbb{Z}_{\geq 0}
   \]

##### Parameters (full listing)

- $P = \{\text{Sedans},\ \text{SUVs},\ \text{Electric Vehicles},\ \text{Hybrid Vehicles},\ \text{Trucks},\ \text{Sports Cars},\ \text{Compact Cars},\ \text{Luxury Sedans},\ \text{Vans},\ \text{Pickup Trucks}\}$
- $W = \{\text{Warehouse 1},\ \text{Warehouse 2},\ \text{Warehouse 3},\ \text{Warehouse 4},\ \text{Warehouse 5},\ \text{Warehouse 6},\ \text{Warehouse 7},\ \text{Warehouse 8},\ \text{Warehouse 9},\ \text{Warehouse 10}\}$
- $\text{Value}_i$ and $\text{Weight}_i$ as listed above for each $i \in P$
- $\text{Capacity}_w$ as listed above for each $w \in W$

##### Model Summary

Maximize total value of cars stored across all warehouses, subject to each warehouse's capacity (in weight units), with integer numbers of each vehicle type per warehouse.