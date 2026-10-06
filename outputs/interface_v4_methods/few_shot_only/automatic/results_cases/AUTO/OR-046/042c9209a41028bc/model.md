#### Abstract Mathematical Model

**Index Sets:**
- $I$: Set of products, indexed by $i$ (from products.csv, column ProductName).

**Parameters:**
- $v_i$: Value (benefit) per unit of product $i$ (from products.csv, column Value).
- $w_i$: Weight (space requirement) per unit of product $i$ (from products.csv, column Weight).
- $C$: Total stock capacity (from capacity.csv, column Capacity).

**Decision Variables:**
- $x_i \in \mathbb{Z}_{\geq 0}$: Number of units of product $i$ to order each day.

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

- $I$: All records in products.csv, column ProductName, in source order.
- $v_i$: products.csv, column Value, for each $i$.
- $w_i$: products.csv, column Weight, for each $i$.
- $C$: capacity.csv, column Capacity, row 0.

Each $x_i$ is the number of units of product $i$ to order each day. The objective maximizes total benefit, subject to the overall stock capacity. All variables are nonnegative integers.