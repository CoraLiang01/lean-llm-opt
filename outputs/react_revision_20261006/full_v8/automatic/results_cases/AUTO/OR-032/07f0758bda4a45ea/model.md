#### Symbolic Mathematical Model

Let:
- $I$ = index set of all products classified under ‘Books’
- For each $i \in I$:
    - $A_i$ = revenue per unit of product $i$
    - $d_i$ = deterministic demand for product $i$
    - $s_i$ = initial inventory for product $i$
    - $x_i$ = number of units of product $i$ to fulfill (decision variable)

Objective:
\[
\max \sum_{i \in I} A_i x_i
\]

Subject to:
\[
\begin{align*}
& x_i \leq d_i && \forall i \in I \quad \text{(Demand constraint)} \\
& x_i \leq s_i && \forall i \in I \quad \text{(Inventory constraint)} \\
& x_i \geq 0 && \forall i \in I \quad \text{(Nonnegativity)} \\
& x_i \in \mathbb{Z} && \forall i \in I \quad \text{(Integer variables, if required)}
\end{align*}
\]

#### Data Mapping

- Index set $I$: All records in table_id file_0_view_0 where Product_Name has prefix "Books"
- Parameter $A_i$: Revenue from column "Revenue" in table_id file_0_view_0
- Parameter $d_i$: Demand from column "Demand" in table_id file_0_view_0
- Parameter $s_i$: Initial Inventory from column "Initial Inventory" in table_id file_0_view_0
- Variable $x_i$: Number of units of product $i$ to fulfill (decision variable for each $i \in I$)