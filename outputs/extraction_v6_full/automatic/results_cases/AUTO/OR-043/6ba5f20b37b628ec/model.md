#### Abstract Mathematical Model

Let:
- $I$ = set of products, indexed by $i$ (with business identifier ProductName from products.csv)
- $x_i$ = number of units of product $i$ to order each day (decision variable, nonnegative integer)

Parameters:
- $b_i$ = benefit per unit of product $i$ (from products.csv, column Value)
- $w_i$ = weight (stock space required) per unit of product $i$ (from products.csv, column Weight)
- $C$ = overall stock capacity (from capacity.csv, column Capacity)

Objective:
$$
\max \sum_{i \in I} b_i x_i
$$

Subject to:
$$
\sum_{i \in I} w_i x_i \leq C
$$

$$
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
$$

---

#### Data Mapping

- $b_i$: products.csv (file_1_view_0), column Value, indexed by ProductName
- $w_i$: products.csv (file_1_view_0), column Weight, indexed by ProductName
- $C$: capacity.csv (file_0_view_0), column Capacity (single value, applies globally)
- $x_i$: decision variable for each ProductName in products.csv

All data is used as returned, preserving original file and row order.