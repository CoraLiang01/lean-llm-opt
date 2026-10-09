#### Mathematical Optimization Model

Let:
- $I$ = set of all products (indexed by $i$)
- $A_i$ = revenue per unit of product $i$
- $d_i$ = demand for product $i$
- $s_i$ = initial inventory for product $i$
- $x_i$ = number of units of product $i$ to fulfill (decision variable)

Objective:
\[
\max \sum_{i \in I} A_i \cdot x_i
\]

Subject to:
\[
\begin{align*}
& x_i \leq d_i && \forall i \in I \quad \text{(Demand constraint)} \\
& x_i \leq s_i && \forall i \in I \quad \text{(Inventory constraint)} \\
& x_i \geq 0 && \forall i \in I \quad \text{(Nonnegativity)} \\
& x_i \in \mathbb{Z} && \forall i \in I \quad \text{(Integer variables)}
\end{align*}
\]

#### Data Mapping

- Table: file_0_view_0 (WomenClothingEcommerceSalesData.csv)
    - Index set $I$: All unique values in column "Product Name"
    - Parameter $A_i$: column "Revenue"
    - Parameter $d_i$: column "Demand"
    - Parameter $s_i$: column "Initial Inventory"
    - Decision variable $x_i$: number of units to fulfill for each $i \in I$