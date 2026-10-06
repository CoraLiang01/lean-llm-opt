#### Abstract Mathematical Model

Let $I$ be the set of products, indexed by $i$ (with business identifier ProductName from products.csv).

Parameters:
- $v_i$: Value per unit of product $i$ (from products.csv, column Value)
- $w_i$: Weight (space requirement) per unit of product $i$ (from products.csv, column Weight)
- $C$: Total stock capacity (from capacity.csv, column Capacity)

Decision variables:
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

#### Data Mapping

- $I$: All ProductName values in file_1_view_0 (products.csv)
- $v_i$: file_1_view_0, column Value, indexed by ProductName
- $w_i$: file_1_view_0, column Weight, indexed by ProductName
- $C$: file_0_view_0, column Capacity

Each $x_i$ is the number of units to order for product $i$ (ProductName from products.csv). The total weight of all ordered products cannot exceed the overall stock capacity $C$ from capacity.csv. The objective is to maximize the total value from the ordered products. All variables are nonnegative integers.