#### Abstract Mathematical Model

**Index Sets:**
- $I$: Set of vehicle types (from products.csv, column ProductName)

**Parameters:**
- $p_i$: Profit per unit of vehicle $i$ (from products.csv, column Value, table_id: file_1_view_0)
- $w_i$: Inventory space required per unit of vehicle $i$ (from products.csv, column Weight, table_id: file_1_view_0)
- $C$: Total inventory capacity (from capacity.csv, column Capacity, table_id: file_0_view_0)

**Decision Variables:**
- $x_i$: Number of vehicles of type $i$ to order per day; $x_i \in \mathbb{Z}_{\geq 0}$

**Objective:**
\[
\max \sum_{i \in I} p_i x_i
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

- $I$: All records in file_1_view_0 (products.csv), column ProductName
- $p_i$: file_1_view_0 (products.csv), column Value, keyed by ProductName
- $w_i$: file_1_view_0 (products.csv), column Weight, keyed by ProductName
- $C$: file_0_view_0 (capacity.csv), column Capacity

All parameters and index sets are to be used exactly as returned, preserving original file and column names.