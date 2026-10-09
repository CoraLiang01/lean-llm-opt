### Mathematical Model

Let $I$ be the set of products from products.csv.

**Decision variables:**
- For each product $i \in I$, let $x_i$ = number of units of product $i$ to order each day ($x_i \in \mathbb{Z}_{\geq 0}$).

**Parameters:**
- $v_i$ = Value of product $i$ (from products.csv, column Value)
- $w_i$ = Weight of product $i$ (from products.csv, column Weight)
- $C$ = Overall stock capacity (from capacity.csv, column Capacity)

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

- $I$: All records in products.csv, column ProductName, table_id file_1_view_0
- $v_i$: products.csv, column Value, table_id file_1_view_0
- $w_i$: products.csv, column Weight, table_id file_1_view_0
- $C$: capacity.csv, column Capacity, table_id file_0_view_0
- $x_i$: Decision variable for each $i \in I$ (product), nonnegative integer

---

**Summary:**  
Maximize total value of ordered products, subject to the overall stock weight capacity. Each product's order quantity is a nonnegative integer. All parameters are mapped directly from the provided CSV files and columns.