### Mathematical Model

Let $I$ be the set of products from products.csv.

**Decision variables:**
- $x_i$: number of units of product $i \in I$ to order each day ($x_i \in \mathbb{Z}_{\geq 0}$)

**Parameters:**
- $v_i$: Value of product $i$ (from products.csv, column Value)
- $w_i$: Weight of product $i$ (from products.csv, column Weight)
- $C$: Overall stock capacity (from capacity.csv, column Capacity)

**Objective:**
\[
\max \sum_{i \in I} v_i x_i
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

- $I$: All records in file_1_view_0 (products.csv), column ProductName
- $v_i$: file_1_view_0, column Value, for each $i \in I$
- $w_i$: file_1_view_0, column Weight, for each $i \in I$
- $C$: file_0_view_0, column Capacity

- $x_i$: number of units of product $i$ to order each day (decision variable, nonnegative integer, indexed by $i \in I$)