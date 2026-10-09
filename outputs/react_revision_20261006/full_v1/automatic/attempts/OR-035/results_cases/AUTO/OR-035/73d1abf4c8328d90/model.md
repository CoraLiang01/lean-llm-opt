#### Symbolic Mathematical Model

Let:
- $I$ = set of bread types (indexed by $i$), corresponding to all ProductName values in products.csv.
- $x_i$ = number of units of bread type $i$ to order each day, $x_i \in \mathbb{Z}_{\geq 0}$

Parameters:
- $p_i$ = expected profit per unit of bread type $i$
- $w_i$ = storage weight per unit of bread type $i$
- $C$ = total storage capacity

Objective:
\[
\max \sum_{i \in I} p_i x_i
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

- $I$: All ProductName values from file_1_view_0 (products.csv, column ProductName)
- $p_i$: file_1_view_0, column Value, mapped by ProductName
- $w_i$: file_1_view_0, column Weight, mapped by ProductName
- $C$: file_0_view_0, column Capacity (capacity.csv, row 0)

Each $x_i$ is the integer number of units to order for bread type $i$ (ProductName). The total storage used, $\sum_{i \in I} w_i x_i$, must not exceed $C$. The objective is to maximize total expected profit.