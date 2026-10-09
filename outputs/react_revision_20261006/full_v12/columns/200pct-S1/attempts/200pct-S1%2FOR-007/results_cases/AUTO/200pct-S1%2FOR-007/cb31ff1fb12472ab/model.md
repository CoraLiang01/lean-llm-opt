### Mathematical Model

Let $I$ be the set of vehicle types (from products.csv, column ProductName).

**Decision variables:**
- For each $i \in I$, let $x_i \in \mathbb{Z}_{\geq 0}$ be the number of vehicles of type $i$ to order per day.

**Parameters:**
- $p_i$: profit per unit of vehicle $i$ (from products.csv, column Value).
- $w_i$: weight (inventory space used) per unit of vehicle $i$ (from products.csv, column Weight).
- $C$: total inventory capacity (from capacity.csv, column Capacity).

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

- $I$: All records in file_1_view_0 (products.csv), column ProductName.
- $p_i$: file_1_view_0, column Value, keyed by ProductName.
- $w_i$: file_1_view_0, column Weight, keyed by ProductName.
- $C$: file_0_view_0, column Capacity (single value).
- $x_i$: Decision variable for each $i \in I$.

**Constraint and objective coefficients are mapped directly from the corresponding columns and rows in the source files.**