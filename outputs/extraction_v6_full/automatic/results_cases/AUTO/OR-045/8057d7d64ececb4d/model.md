#### Abstract Mathematical Model

Let $I$ be the set of produce types, indexed by $i$ (with business identifier ProductName from products.csv).

Parameters:
- $b_i$: benefit per unit of produce $i$ (from products.csv, column Value)
- $w_i$: weight per unit of produce $i$ (from products.csv, column Weight)
- $C$: total inventory capacity (from capacity.csv, column Capacity)

Decision variables:
- $x_i$: number of units of produce $i$ to order daily ($x_i \in \mathbb{Z}_{\geq 0}$)

Objective:
\[
\max \sum_{i \in I} b_i x_i
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
- $b_i$: file_1_view_0, column Value, indexed by ProductName
- $w_i$: file_1_view_0, column Weight, indexed by ProductName
- $C$: file_0_view_0, column Capacity

Each $x_i$ is the integer number of units to order for produce $i$ (ProductName). The total weight of all ordered units cannot exceed $C$. The objective is to maximize total benefit.