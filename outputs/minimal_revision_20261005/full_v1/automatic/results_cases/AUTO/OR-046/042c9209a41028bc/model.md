ABSTRACT MATHEMATICAL MODEL

Sets:
- $I$: Set of products, indexed by $i$ (from products.csv, column ProductName)

Parameters:
- $v_i$: Value (benefit) per unit of product $i$ (from products.csv, column Value)
- $w_i$: Weight (stock space required) per unit of product $i$ (from products.csv, column Weight)
- $C$: Overall stock capacity (from capacity.csv, column Capacity)

Decision Variables:
- $x_i$: Number of units of product $i$ to order each day ($x_i \in \mathbb{Z}_{\geq 0}$)

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

DATA MAPPING

- $I$: All records in file_1_view_0 (products.csv), column ProductName
- $v_i$: file_1_view_0 (products.csv), column Value, keyed by ProductName
- $w_i$: file_1_view_0 (products.csv), column Weight, keyed by ProductName
- $C$: file_0_view_0 (capacity.csv), column Capacity

All parameters and sets are defined exactly as returned by CSVQA, preserving file and column names.