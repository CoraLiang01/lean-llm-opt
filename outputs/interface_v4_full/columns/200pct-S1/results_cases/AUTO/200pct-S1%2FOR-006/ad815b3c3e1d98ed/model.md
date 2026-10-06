#### Abstract Mathematical Model

**Index Sets:**
- $I$: Set of vehicle types, indexed by $i$ (from products.csv, column ProductName).

**Parameters:**
- $v_i$: Benefit coefficient of vehicle type $i$ (from products.csv, column Value, table_id: file_1_view_0).
- $w_i$: Inventory weight (space requirement) of vehicle type $i$ (from products.csv, column Weight, table_id: file_1_view_0).
- $C$: Total inventory capacity (from capacity.csv, column Capacity, table_id: file_0_view_0).

**Decision Variables:**
- $x_i$: Number of units of vehicle type $i$ to order daily. ($x_i \in \mathbb{Z}_{\geq 0}$, integer and nonnegative)

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

#### Data Mapping

- $I$: All records in products.csv, table_id: file_1_view_0, column ProductName.
- $v_i$: products.csv, table_id: file_1_view_0, column Value, keyed by ProductName.
- $w_i$: products.csv, table_id: file_1_view_0, column Weight, keyed by ProductName.
- $C$: capacity.csv, table_id: file_0_view_0, column Capacity.

All parameters and index sets are to be used exactly as returned, preserving original file and row order.