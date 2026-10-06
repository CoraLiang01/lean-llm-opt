**Abstract Mathematical Model**

**Index Sets**
- $I$: Set of products, indexed by $i$. (from products.csv, column ProductName)

**Parameters**
- $v_i$: Value (benefit) per unit of product $i$. (from products.csv, column Value)
- $w_i$: Weight (stock space required) per unit of product $i$. (from products.csv, column Weight)
- $C$: Total stock capacity. (from capacity.csv, column Capacity)

**Decision Variables**
- $x_i \in \mathbb{Z}_{\geq 0}$: Number of units of product $i$ to order each day.

**Objective**
\[
\max \sum_{i \in I} v_i x_i
\]

**Constraints**
\[
\sum_{i \in I} w_i x_i \leq C
\]
\[
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
\]

---

**Data Mapping**

- $I$: All records in file_1_view_0 (products.csv), column ProductName.
- $v_i$: file_1_view_0, column Value, keyed by ProductName.
- $w_i$: file_1_view_0, column Weight, keyed by ProductName.
- $C$: file_0_view_0, column Capacity, the unique returned value.
- $x_i$: Decision variable for each $i \in I$.

---

**Summary:**  
Choose integer quantities $x_i$ of each product $i$ to maximize total value, subject to the overall stock capacity. All parameters and index sets are mapped directly to the supplied data columns and rows.