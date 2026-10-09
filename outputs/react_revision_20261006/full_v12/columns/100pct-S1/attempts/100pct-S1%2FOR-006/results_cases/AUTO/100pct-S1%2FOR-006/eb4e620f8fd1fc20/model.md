### Mathematical Model

Let $I$ be the set of vehicle types, indexed by $i$ (with identifiers ProductName from products.csv).

**Parameters:**
- $v_i$: Value (benefit) of vehicle type $i$ (from products.csv, column Value)
- $w_i$: Weight (inventory space required) of vehicle type $i$ (from products.csv, column Weight)
- $C$: Total inventory capacity (from capacity.csv, column Capacity)

**Decision Variables:**
- $x_i$: Number of units of vehicle type $i$ to order daily; $x_i \in \mathbb{Z}_{\geq 0}$

**Objective:**
\[
\max \sum_{i \in I} v_i x_i
\]

**Constraints:**
\[
\sum_{i \in I} w_i x_i \leq C
\]
\[
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
\]

---

### Data Mapping

- $I$: All records in products.csv, column ProductName (table_id: file_1_view_0, column: ProductName)
- $v_i$: products.csv, column Value (table_id: file_1_view_0, column: Value)
- $w_i$: products.csv, column Weight (table_id: file_1_view_0, column: Weight)
- $C$: capacity.csv, column Capacity (table_id: file_0_view_0, column: Capacity; single record applies globally)
- $x_i$: Decision variable for each $i \in I$ (vehicle type/ProductName)

All parameters and index sets are mapped directly from the current CSV data as described above.