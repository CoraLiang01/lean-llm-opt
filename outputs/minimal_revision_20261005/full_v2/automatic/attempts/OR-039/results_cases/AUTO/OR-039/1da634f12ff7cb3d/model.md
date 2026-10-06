**Abstract Mathematical Model**

**Index Sets:**
- $I$: Set of vehicle types (from products.csv, column ProductName)
- $J$: Set of warehouses (from capacity.csv, column Warehouse ID)

**Parameters:**
- $v_i$: Value (benefit coefficient) of vehicle type $i$ (from products.csv, column Value)
- $w_i$: Weight (space requirement) of vehicle type $i$ (from products.csv, column Weight)
- $C_j$: Capacity of warehouse $j$ (from capacity.csv, column Capacity)

**Decision Variables:**
- $x_{ij}$: Number of units of vehicle type $i$ to store in warehouse $j$  
  Domain: $x_{ij} \in \mathbb{Z}_{\geq 0}$ (nonnegative integers)

---

**Objective:**
\[
\max \sum_{j \in J} \sum_{i \in I} v_i \cdot x_{ij}
\]

**Subject to:**

1. **Warehouse Capacity Constraints:**  
   For each warehouse $j \in J$,
   \[
   \sum_{i \in I} w_i \cdot x_{ij} \leq C_j
   \]

2. **Nonnegativity and Integrality:**  
   For all $i \in I$, $j \in J$,
   \[
   x_{ij} \in \mathbb{Z}_{\geq 0}
   \]

---

**Data Mapping**

- $I$ (vehicle types): file_1_view_0, column ProductName
- $J$ (warehouses): file_0_view_0, column Warehouse ID
- $v_i$: file_1_view_0, column Value, keyed by ProductName
- $w_i$: file_1_view_0, column Weight, keyed by ProductName
- $C_j$: file_0_view_0, column Capacity, keyed by Warehouse ID

---

**Notes:**
- All vehicle types and warehouses from the source files are included.
- Each $x_{ij}$ represents the number of units of vehicle type $i$ stored in warehouse $j$.
- All variables are nonnegative integers as required.