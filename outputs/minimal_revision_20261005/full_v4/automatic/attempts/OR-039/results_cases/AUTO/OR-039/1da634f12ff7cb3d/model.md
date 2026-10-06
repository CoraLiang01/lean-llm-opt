**Abstract Mathematical Model**

**Index Sets:**
- $I$: Set of vehicle types (from products.csv, column ProductName)
- $J$: Set of warehouses (from capacity.csv, column Warehouse ID)

**Parameters:**
- $v_i$: Value (benefit coefficient) of vehicle type $i$ (from products.csv, column Value, indexed by ProductName)
- $w_i$: Weight (space requirement) of vehicle type $i$ (from products.csv, column Weight, indexed by ProductName)
- $C_j$: Capacity of warehouse $j$ (from capacity.csv, column Capacity, indexed by Warehouse ID)

**Decision Variables:**
- $x_{ij}$: Number of units of vehicle type $i$ to store in warehouse $j$; $x_{ij} \in \mathbb{Z}_{\geq 0}$

**Objective:**
\[
\max \sum_{i \in I} \sum_{j \in J} v_i \cdot x_{ij}
\]

**Constraints:**
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
- $v_i$: file_1_view_0, column Value, indexed by ProductName
- $w_i$: file_1_view_0, column Weight, indexed by ProductName
- $C_j$: file_0_view_0, column Capacity, indexed by Warehouse ID

---

**Notes:**
- All vehicle types and warehouses from the returned data are included.
- Each $x_{ij}$ is a nonnegative integer, representing the number of units of vehicle type $i$ stored in warehouse $j$.
- All parameters are mapped directly to the columns and business IDs as specified in the data.