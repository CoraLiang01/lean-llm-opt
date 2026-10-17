#### Abstract Mathematical Model

Let:

- $I$: Index set of all products classified as ‘Aalop’.
- For each $i \in I$:
    - $A_i$: Revenue per unit of product $i$ (from column ‘Revenue’).
    - $d_i$: Demand for product $i$ over the sales horizon (from column ‘Demand’).
    - $s_i$: Initial inventory of product $i$ (from column ‘Initial Inventory’).
    - $x_i$: Number of units of product $i$ to fulfill (decision variable).

Objective:
\[
\max \sum_{i \in I} A_i \cdot x_i
\]

Subject to:
\[
\begin{align*}
& x_i \leq d_i, \quad \forall i \in I \quad \text{(Demand constraint)} \\
& x_i \leq s_i, \quad \forall i \in I \quad \text{(Inventory constraint)} \\
& x_i \geq 0, \quad \forall i \in I \quad \text{(Non-negativity, continuous or integer as appropriate)}
\end{align*}
\]

#### Data Mapping

- Table: file_0_view_0 (RestaurantSalesreport.csv)
    - Product Name: Used to identify products classified as ‘Aalop’ (apply query-specific subset).
    - Revenue: Parameter $A_i$.
    - Demand: Parameter $d_i$.
    - Initial Inventory: Parameter $s_i$.

Only products classified as ‘Aalop’ are included in the index set $I$ and the model. All parameters are mapped from the specified columns in the table.