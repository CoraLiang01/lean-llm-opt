#### Mathematical Optimization Model

Let:
- $\mathcal{B}$ = set of all products classified under ‘Books’ (indexed by $i$)
- $A_i$ = revenue per unit of product $i \in \mathcal{B}$
- $d_i$ = deterministic demand for product $i \in \mathcal{B}$
- $I_i$ = initial inventory for product $i \in \mathcal{B}$
- $x_i$ = number of units of product $i$ to fulfill (decision variable), $x_i \in \mathbb{Z}_+$

Objective:
\[
\max \sum_{i \in \mathcal{B}} A_i x_i
\]

Subject to:
\[
x_i \leq d_i \qquad \forall i \in \mathcal{B}
\]
\[
x_i \leq I_i \qquad \forall i \in \mathcal{B}
\]
\[
x_i \geq 0,\quad x_i \in \mathbb{Z} \qquad \forall i \in \mathcal{B}
\]

#### Data Mapping

- Index set $\mathcal{B}$: All rows in table_id file_0_view_0 where Product_Name has prefix "Books"
- Parameter $A_i$: column "Revenue" in table_id file_0_view_0
- Parameter $d_i$: column "Demand" in table_id file_0_view_0
- Parameter $I_i$: column "Initial Inventory" in table_id file_0_view_0
- Variable $x_i$: defined for each $i \in \mathcal{B}$

All data is sourced from table_id file_0_view_0, columns: Product_Name, Revenue, Demand, Initial Inventory.