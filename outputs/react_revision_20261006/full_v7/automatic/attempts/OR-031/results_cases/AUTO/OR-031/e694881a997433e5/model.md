#### Mathematical Optimization Model

Let:
- $I$ = set of all dairy products (indexed by $i$), as defined by the Full_Product_Name column.
- For each $i \in I$:
    - $A_i$ = revenue per unit of product $i$ (from Revenue column)
    - $d_i$ = deterministic demand for product $i$ (from Demand column)
    - $s_i$ = initial inventory for product $i$ (from Initial Inventory column)
    - $x_i$ = number of units of product $i$ to fulfill (decision variable)

Objective:
\[
\max \sum_{i \in I} A_i x_i
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

- Table: file_0_view_0 (DairyGoodsSalesDataset.csv)
    - Index set $I$: Full_Product_Name
    - Parameter $A_i$: Revenue
    - Parameter $d_i$: Demand
    - Parameter $s_i$: Initial Inventory