Let $x_{ij}$ be the number of units of vehicle type $i$ to be stored in warehouse $j$. All $x_{ij}$ are nonnegative integers.

Let $I$ be the set of vehicle types (indexed by ProductName), and $J$ be the set of warehouses (indexed by Warehouse ID).

Let $v_i$ be the Value of vehicle type $i$, and $w_i$ be the Weight of vehicle type $i$ (space required per unit). Let $C_j$ be the Capacity of warehouse $j$.

---

**Sets:**

- $I =$ {Sedans, SUVs, Electric Vehicles, Hybrid Vehicles, Trucks, Sports Cars, Compact Cars, Luxury Sedans, Vans, Pickup Trucks}
- $J =$ {Warehouse 1, Warehouse 2, Warehouse 3, Warehouse 4, Warehouse 5, Warehouse 6, Warehouse 7, Warehouse 8, Warehouse 9, Warehouse 10}

**Parameters:**

- $v_i$ (Value):  
  Sedans: 1200  
  SUVs: 1800  
  Electric Vehicles: 2500  
  Hybrid Vehicles: 2000  
  Trucks: 1500  
  Sports Cars: 3000  
  Compact Cars: 1000  
  Luxury Sedans: 3500  
  Vans: 1600  
  Pickup Trucks: 1700

- $w_i$ (Weight):  
  Sedans: 20  
  SUVs: 15  
  Electric Vehicles: 25  
  Hybrid Vehicles: 18  
  Trucks: 10  
  Sports Cars: 5  
  Compact Cars: 22  
  Luxury Sedans: 8  
  Vans: 12  
  Pickup Trucks: 7

- $C_j$ (Capacity):  
  Warehouse 1: 100  
  Warehouse 2: 80  
  Warehouse 3: 120  
  Warehouse 4: 90  
  Warehouse 5: 50  
  Warehouse 6: 30  
  Warehouse 7: 110  
  Warehouse 8: 40  
  Warehouse 9: 60  
  Warehouse 10: 35

---

**Decision Variables:**

- $x_{ij} \in \mathbb{Z}_{\geq 0}$, for all $i \in I$, $j \in J$

---

**Objective:**

$$
\max \sum_{i \in I} \sum_{j \in J} v_i \cdot x_{ij}
$$

---

**Constraints:**

For each warehouse $j \in J$:
$$
\sum_{i \in I} w_i \cdot x_{ij} \leq C_j
$$

For all $i \in I$, $j \in J$:
$$
x_{ij} \in \mathbb{Z}_{\geq 0}
$$

---

**Data Used (in source order):**

- Warehouses and capacities:
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

- Vehicle types, values, and weights:
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