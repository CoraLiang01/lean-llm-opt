#### Mathematical Model

Let $I$ be the set of vehicle types, indexed by $i$ (from all ProductName in products.csv).

Parameters:
- $v_i$: Value of vehicle type $i$ (from Value in products.csv, table_id file_1_view_0)
- $w_i$: Weight (inventory space required) of vehicle type $i$ (from Weight in products.csv, table_id file_1_view_0)
- $C$: Total inventory capacity (from Capacity in capacity.csv, table_id file_0_view_0)

Decision variables:
- $x_i$: Number of units of vehicle type $i$ to order daily ($x_i \in \mathbb{Z}_{\geq 0}$)

Objective:
\[
\max \sum_{i \in I} v_i x_i
\]

Subject to:
\[
\sum_{i \in I} w_i x_i \leq C
\]
\[
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
\]

---

#### Data Mapping

- $I$: All ProductName in products.csv (table_id file_1_view_0, column ProductName)
- $v_i$: Value from products.csv (table_id file_1_view_0, column Value, keyed by ProductName)
- $w_i$: Weight from products.csv (table_id file_1_view_0, column Weight, keyed by ProductName)
- $C$: Capacity from capacity.csv (table_id file_0_view_0, column Capacity)

All variables, parameters, and constraints are mapped directly to the columns and rows as specified above.