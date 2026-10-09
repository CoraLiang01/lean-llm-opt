##### Sets and Indices

- Let $P$ be the set of vehicle types (products), indexed by $i$.
- Let $W$ be the set of warehouses, indexed by $k$.

##### Parameters

From products.csv (in source order):

| $i$                | Value $v_i$ | Weight $w_i$ |
|--------------------|-------------|--------------|
| Sedans             | 1200        | 20           |
| SUVs               | 1800        | 15           |
| Electric Vehicles  | 2500        | 25           |
| Hybrid Vehicles    | 2000        | 18           |
| Trucks             | 1500        | 10           |
| Sports Cars        | 3000        | 5            |
| Compact Cars       | 1000        | 22           |
| Luxury Sedans      | 3500        | 8            |
| Vans               | 1600        | 12           |
| Pickup Trucks      | 1700        | 7            |

From capacity.csv (in source order):

| $k$           | Capacity $C_k$ |
|---------------|----------------|
| Warehouse 1   | 100            |
| Warehouse 2   | 80             |
| Warehouse 3   | 120            |
| Warehouse 4   | 90             |
| Warehouse 5   | 50             |
| Warehouse 6   | 30             |
| Warehouse 7   | 110            |
| Warehouse 8   | 40             |
| Warehouse 9   | 60             |
| Warehouse 10  | 35             |

##### Decision Variables

- $x_{ik} \in \mathbb{Z}_{\geq 0}$: Number of vehicles of type $i$ to be stored in warehouse $k$.

##### Objective Function

$$
\max \sum_{k \in W} \sum_{i \in P} v_i x_{ik}
$$

##### Constraints

1. **Warehouse Capacity Constraints** (for each warehouse $k$):

   $$
   \sum_{i \in P} w_i x_{ik} \leq C_k \qquad \forall k \in W
   $$

2. **Non-negativity and Integrality**:

   $$
   x_{ik} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in P,\, k \in W
   $$

---

###### Retrieved Information

- Warehouses and capacities (in source order):

  - Warehouse 1: 100
  - Warehouse 2: 80
  - Warehouse 3: 120
  - Warehouse 4: 90
  - Warehouse 5: 50
  - Warehouse 6: 30
  - Warehouse 7: 110
  - Warehouse 8: 40
  - Warehouse 9: 60
  - Warehouse 10: 35

- Products, values, and weights (in source order):

  - Sedans: Value 1200, Weight 20
  - SUVs: Value 1800, Weight 15
  - Electric Vehicles: Value 2500, Weight 25
  - Hybrid Vehicles: Value 2000, Weight 18
  - Trucks: Value 1500, Weight 10
  - Sports Cars: Value 3000, Weight 5
  - Compact Cars: Value 1000, Weight 22
  - Luxury Sedans: Value 3500, Weight 8
  - Vans: Value 1600, Weight 12
  - Pickup Trucks: Value 1700, Weight 7