### Mathematical Model

Let $I$ be the set of vehicle types (from ProductName in products.csv).

**Decision variables:**
- $x_i$: Number of vehicles of type $i \in I$ to order per day. ($x_i \in \mathbb{Z}_{\geq 0}$)

**Parameters:**
- $p_i$: Profit per unit of vehicle $i$ (Value from products.csv)
- $w_i$: Weight (inventory space) per unit of vehicle $i$ (Weight from products.csv)
- $C$: Total inventory capacity (Capacity from capacity.csv)

**Objective:**
\[
\max \sum_{i \in I} p_i x_i
\]

**Constraint:**
\[
\sum_{i \in I} w_i x_i \leq C
\]
\[
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
\]

---

### Data Mapping

- $I$: All ProductName in file_1_view_0 (products.csv)
- $p_i$: Value column in file_1_view_0, keyed by ProductName
- $w_i$: Weight column in file_1_view_0, keyed by ProductName
- $C$: Capacity column in file_0_view_0 (capacity.csv), single value

All parameters and index sets are mapped directly from the returned CSV data.