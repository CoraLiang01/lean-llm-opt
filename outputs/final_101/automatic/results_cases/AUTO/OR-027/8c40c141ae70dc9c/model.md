#### Abstract Mathematical Model

Let:
- $I$ = index set of all products classified as ‘Organ’
- For each $i \in I$:
    - $A_i$ = revenue per unit of product $i$
    - $d_i$ = deterministic demand for product $i$
    - $S_i$ = initial inventory of product $i$
    - $x_i$ = number of units of product $i$ to fulfill (decision variable)

Objective:
\[
\max \sum_{i \in I} A_i \cdot x_i
\]

Subject to:
\[
\begin{align*}
& x_i \leq d_i, \quad \forall i \in I \\
& x_i \leq S_i, \quad \forall i \in I \\
& x_i \in \mathbb{Z}_+, \quad \forall i \in I
\end{align*}
\]

#### Data Mapping

- Table: SupermartGrocerySales-RetailAnalyticsDataset.csv
    - Index set $I$: All rows where [Sub Category] contains or starts with "Organ"
    - Parameter $A_i$: [Revenue] column
    - Parameter $d_i$: [Demand] column
    - Parameter $S_i$: [Initial Inventory] column
    - Product identifier: [Sub Category] column

No additional constraints or data transformations are imposed beyond those specified above.