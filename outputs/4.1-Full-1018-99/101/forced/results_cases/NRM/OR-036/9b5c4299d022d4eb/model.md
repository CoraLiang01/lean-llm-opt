#### Abstract Mathematical Model

**Index Sets:**
- $I$: Set of vehicle types (from products.csv, column ProductName).

**Parameters:**
- $v_i$: Benefit coefficient for vehicle type $i \in I$ (from products.csv, column Value).
- $w_i$: Inventory weight per unit for vehicle type $i \in I$ (from products.csv, column Weight).
- $C$: Total inventory capacity (from capacity.csv, column Capacity).

**Decision Variables:**
- $x_i$: Number of units of vehicle type $i$ to order daily, $x_i \in \mathbb{Z}_{\geq 0}$, $\forall i \in I$.

**Objective:**
\[
\max \sum_{i \in I} v_i \cdot x_i
\]

**Constraints:**
1. **Total Inventory Capacity:**
   \[
   \sum_{i \in I} w_i \cdot x_i \leq C
   \]
2. **Integer and Nonnegativity:**
   \[
   x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i \in I
   \]

---

#### Data Mapping

- $I$: All records in products.csv, column ProductName (table_id: file_1_view_0, column: ProductName)
- $v_i$: products.csv, column Value (table_id: file_1_view_0, column: Value)
- $w_i$: products.csv, column Weight (table_id: file_1_view_0, column: Weight)
- $C$: capacity.csv, column Capacity (table_id: file_0_view_0, column: Capacity)