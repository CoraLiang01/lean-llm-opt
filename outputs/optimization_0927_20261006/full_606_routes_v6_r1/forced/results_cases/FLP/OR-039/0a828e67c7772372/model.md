##### Sets and Indices

- Let $P$ be the set of vehicle types (products):  
  $P = \{\text{Sedans},\ \text{SUVs},\ \text{Electric Vehicles},\ \text{Hybrid Vehicles},\ \text{Trucks},\ \text{Sports Cars},\ \text{Compact Cars},\ \text{Luxury Sedans},\ \text{Vans},\ \text{Pickup Trucks}\}$

- Let $W$ be the set of warehouses:  
  $W = \{\text{Warehouse 1},\ \text{Warehouse 2},\ \text{Warehouse 3},\ \text{Warehouse 4},\ \text{Warehouse 5},\ \text{Warehouse 6},\ \text{Warehouse 7},\ \text{Warehouse 8},\ \text{Warehouse 9},\ \text{Warehouse 10}\}$

##### Parameters

- Value per unit for each product $i \in P$:
  - $\text{Value}_{\text{Sedans}} = 1200$
  - $\text{Value}_{\text{SUVs}} = 1800$
  - $\text{Value}_{\text{Electric Vehicles}} = 2500$
  - $\text{Value}_{\text{Hybrid Vehicles}} = 2000$
  - $\text{Value}_{\text{Trucks}} = 1500$
  - $\text{Value}_{\text{Sports Cars}} = 3000$
  - $\text{Value}_{\text{Compact Cars}} = 1000$
  - $\text{Value}_{\text{Luxury Sedans}} = 3500$
  - $\text{Value}_{\text{Vans}} = 1600$
  - $\text{Value}_{\text{Pickup Trucks}} = 1700$

- Weight per unit for each product $i \in P$:
  - $\text{Weight}_{\text{Sedans}} = 20$
  - $\text{Weight}_{\text{SUVs}} = 15$
  - $\text{Weight}_{\text{Electric Vehicles}} = 25$
  - $\text{Weight}_{\text{Hybrid Vehicles}} = 18$
  - $\text{Weight}_{\text{Trucks}} = 10$
  - $\text{Weight}_{\text{Sports Cars}} = 5$
  - $\text{Weight}_{\text{Compact Cars}} = 22$
  - $\text{Weight}_{\text{Luxury Sedans}} = 8$
  - $\text{Weight}_{\text{Vans}} = 12$
  - $\text{Weight}_{\text{Pickup Trucks}} = 7$

- Capacity for each warehouse $w \in W$:
  - $\text{Capacity}_{\text{Warehouse 1}} = 100$
  - $\text{Capacity}_{\text{Warehouse 2}} = 80$
  - $\text{Capacity}_{\text{Warehouse 3}} = 120$
  - $\text{Capacity}_{\text{Warehouse 4}} = 90$
  - $\text{Capacity}_{\text{Warehouse 5}} = 50$
  - $\text{Capacity}_{\text{Warehouse 6}} = 30$
  - $\text{Capacity}_{\text{Warehouse 7}} = 110$
  - $\text{Capacity}_{\text{Warehouse 8}} = 40$
  - $\text{Capacity}_{\text{Warehouse 9}} = 60$
  - $\text{Capacity}_{\text{Warehouse 10}} = 35$

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

##### Parameters (full list):

- $P = \{\text{Sedans},\ \text{SUVs},\ \text{Electric Vehicles},\ \text{Hybrid Vehicles},\ \text{Trucks},\ \text{Sports Cars},\ \text{Compact Cars},\ \text{Luxury Sedans},\ \text{Vans},\ \text{Pickup Trucks}\}$
- $W = \{\text{Warehouse 1},\ \text{Warehouse 2},\ \text{Warehouse 3},\ \text{Warehouse 4},\ \text{Warehouse 5},\ \text{Warehouse 6},\ \text{Warehouse 7},\ \text{Warehouse 8},\ \text{Warehouse 9},\ \text{Warehouse 10}\}$
- $\text{Value}_i$ and $\text{Weight}_i$ as listed above for each $i \in P$
- $\text{Capacity}_w$ as listed above for each $w \in W$

##### Model Summary

Maximize total value of cars stored, subject to each warehouse's capacity, with integer numbers of each vehicle type per warehouse.