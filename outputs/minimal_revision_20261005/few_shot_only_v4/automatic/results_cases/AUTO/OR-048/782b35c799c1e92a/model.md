**Abstract Mathematical Model**

**Index Sets:**
- $I$: Set of storage areas (indexed by $i$), from capacity.csv [StorageID]
- $J$: Set of air conditioner types (indexed by $j$), from products.csv [ProductName]

**Parameters:**
- $c_i$: Capacity of storage area $i$ (from capacity.csv [Capacity])
- $v_j$: Value of air conditioner type $j$ (from products.csv [Value])
- $w_j$: Weight (size) of air conditioner type $j$ (from products.csv [Weight])

**Decision Variables:**
- $x_{ij}$: Number of units of air conditioner type $j$ to place in storage area $i$; $x_{ij} \in \mathbb{Z}_{\geq 0}$

---

**Objective:**
\[
\max \sum_{i \in I} \sum_{j \in J} v_j \cdot x_{ij}
\]

**Subject to:**

1. **Storage Area Capacity Constraints:**
   \[
   \sum_{j \in J} w_j \cdot x_{ij} \leq c_i \qquad \forall i \in I
   \]

2. **Integrality and Nonnegativity:**
   \[
   x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in I,\, j \in J
   \]

---

**Data Mapping**

- $I$ (storage areas): /.../capacity.csv, column [StorageID]
- $c_i$: /.../capacity.csv, column [Capacity], key [StorageID]
- $J$ (air conditioner types): /.../products.csv, column [ProductName]
- $v_j$: /.../products.csv, column [Value], key [ProductName]
- $w_j$: /.../products.csv, column [Weight], key [ProductName]