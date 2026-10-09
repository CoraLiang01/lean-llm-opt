#### Mathematical Optimization Model

Let:
- $I$ = set of all products classified under ‘27in’ (from the data, indexed by $i$)
- For each $i \in I$:
    - $A_i$ = revenue per unit of product $i$
    - $d_i$ = demand for product $i$
    - $s_i$ = initial inventory of product $i$
    - $x_i$ = number of units of product $i$ to fulfill (decision variable)

Objective:
\[
\max \sum_{i \in I} A_i \cdot x_i
\]

Subject to:
\[
x_i \leq d_i \quad \forall i \in I
\]
\[
x_i \leq s_i \quad \forall i \in I
\]
\[
x_i \geq 0 \quad \forall i \in I
\]
\[
x_i \in \mathbb{Z} \quad \forall i \in I
\]

#### Data Mapping

- Table: file_0_view_0 (SalesDataAnalysis.csv)
    - Index set $I$: All rows where Product Name starts with "27in"
    - Parameter $A_i$: column "Revenue"
    - Parameter $d_i$: column "Demand"
    - Parameter $s_i$: column "Initial Inventory"
    - Variable $x_i$: number of units to fulfill for each $i \in I$