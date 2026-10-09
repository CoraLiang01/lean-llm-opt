#### Symbolic Mathematical Model

Let:
- $I$ = set of all dairy products (indexed by $i$)
- $A_i$ = revenue per unit of product $i$
- $d_i$ = deterministic demand for product $i$
- $s_i$ = initial inventory for product $i$
- $x_i$ = number of units of product $i$ to fulfill (decision variable)

Objective:
\[
\max \sum_{i \in I} A_i \cdot x_i
\]

Subject to:
\[
\begin{align*}
& x_i \leq d_i, \quad \forall i \in I \\
& x_i \leq s_i, \quad \forall i \in I \\
& x_i \geq 0, \quad x_i \in \mathbb{Z}, \quad \forall i \in I
\end{align*}
\]

#### Data Mapping

- Index set $I$: All unique values in column Full_Product_Name from table_id file_0_view_0 (DairyGoodsSalesDataset.csv)
- Parameter $A_i$: Revenue from column Revenue in table_id file_0_view_0
- Parameter $d_i$: Demand from column Demand in table_id file_0_view_0
- Parameter $s_i$: Initial Inventory from column Initial Inventory in table_id file_0_view_0
- Decision variable $x_i$: Number of units fulfilled for each $i \in I$ (as defined above)