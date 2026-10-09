##### Mathematical Model

Let $I$ be the set of vehicle types, indexed by $i$ (from all ProductName in file_1_view_0).

**Parameters:**
- $v_i$: Value (benefit coefficient) of vehicle type $i$ (from Value in file_1_view_0)
- $w_i$: Weight (inventory space required) of vehicle type $i$ (from Weight in file_1_view_0)
- $C$: Total inventory capacity (from Capacity in file_0_view_0)

**Decision Variables:**
- $x_i$: Number of units of vehicle type $i$ to order daily; $x_i \in \mathbb{Z}_{\geq 0}$

**Objective:**
\[
\max \sum_{i \in I} v_i x_i
\]

**Subject to:**
\[
\sum_{i \in I} w_i x_i \leq C
\]
\[
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
\]

---

##### Data Mapping

- $I$: All records ProductName in file_1_view_0 (products.csv)
- $v_i$: Value column in file_1_view_0, keyed by ProductName
- $w_i$: Weight column in file_1_view_0, keyed by ProductName
- $C$: Capacity column in file_0_view_0 (capacity.csv), single value
- $x_i$: Decision variable for each ProductName in file_1_view_0

All parameters and index sets are mapped directly from the returned tables, preserving original identifiers.