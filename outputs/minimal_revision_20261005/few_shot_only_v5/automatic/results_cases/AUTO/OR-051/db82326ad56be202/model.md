**Abstract Mathematical Model**

**Index Sets:**
- $C$: Set of cabinets, indexed by $i$ (from capacity.csv, CabinetID)
- $P$: Set of coffee products, indexed by $j$ (from products.csv, ProductName)

**Parameters:**
- $v_j$: Value per unit of product $j$ (from products.csv, Value)
- $w_j$: Weight per unit of product $j$ (from products.csv, Weight)
- $cap_i$: Capacity of cabinet $i$ (from capacity.csv, Capacity)

**Decision Variables:**
- $x_{ij}$: Number of units of product $j$ to place in cabinet $i$; $x_{ij} \in \mathbb{Z}_{\geq 0}$

---

**Objective:**
\[
\max \sum_{i \in C} \sum_{j \in P} v_j \, x_{ij}
\]

**Constraints:**

1. **Cabinet Capacity Constraints:**  
   For each cabinet $i \in C$,
   \[
   \sum_{j \in P} w_j \, x_{ij} \leq cap_i
   \]

2. **Integrality and Nonnegativity:**  
   For all $i \in C$, $j \in P$,
   \[
   x_{ij} \in \mathbb{Z}_{\geq 0}
   \]

---

**Data Mapping**

| Symbol      | Source Table & Column                                      | Index Set / Key         |
|-------------|------------------------------------------------------------|-------------------------|
| $C$         | capacity.csv, CabinetID                                    | CabinetID               |
| $P$         | products.csv, ProductName                                  | ProductName             |
| $v_j$       | products.csv, Value                                        | ProductName             |
| $w_j$       | products.csv, Weight                                       | ProductName             |
| $cap_i$     | capacity.csv, Capacity                                     | CabinetID               |
| $x_{ij}$    | Decision variable: units of product $j$ in cabinet $i$     | CabinetID, ProductName  |