#### Symbolic Mathematical Model

Let:
- $I$ = index set of all products with $id\_number$ prefix "id999" (from the data)
- For each $i \in I$:
    - $A_i$ = revenue per unit of product $i$ (parameter)
    - $d_i$ = demand for product $i$ during the sales horizon (parameter)
    - $s_i$ = initial inventory of product $i$ (parameter)
    - $x_i$ = number of units of product $i$ to fulfill (decision variable, integer, $x_i \geq 0$)

Objective:
\[
\max \sum_{i \in I} A_i \cdot x_i
\]

Subject to:
\[
\begin{align*}
& x_i \leq s_i, \quad \forall i \in I \quad \text{(Inventory constraint)} \\
& x_i \leq d_i, \quad \forall i \in I \quad \text{(Demand constraint)} \\
& x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i \in I \quad \text{(Non-negative integer variables)}
\end{align*}
\]

#### Data Mapping

- Table: file_0_view_0 (from OnlineRetailSalesDataset.csv)
    - Index set $I$: All rows where column "id_number" has prefix "id999"
    - Parameter $A_i$: column "Revenue"
    - Parameter $d_i$: column "Demand"
    - Parameter $s_i$: column "Initial Inventory"
    - Variable $x_i$: fulfillment quantity for product $i$ (decision variable as defined above)