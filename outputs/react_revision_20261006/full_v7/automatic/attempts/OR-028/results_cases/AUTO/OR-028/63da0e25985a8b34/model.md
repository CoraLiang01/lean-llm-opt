#### Mathematical Optimization Model

Let:
- $I$ = set of all products (indexed by $i$), as defined by the "Product Name" column.

Parameters (for each $i \in I$):
- $A_i$ = revenue per unit of product $i$ ("Revenue" column)
- $d_i$ = total demand for product $i$ ("Demand" column)
- $s_i$ = initial inventory for product $i$ ("Initial Inventory" column)

Decision variables:
- $x_i$ = number of units of product $i$ to fulfill, $x_i \in \mathbb{Z}_+$

Objective:
\[
\max \sum_{i \in I} A_i x_i
\]

Subject to:
\[
\begin{align*}
& x_i \leq d_i && \forall i \in I \quad \text{(Demand constraint)} \\
& x_i \leq s_i && \forall i \in I \quad \text{(Inventory constraint)} \\
& x_i \geq 0 && \forall i \in I \quad \text{(Nonnegativity and integrality)}
\end{align*}
\]

#### Data Mapping

- Table: file_0_view_0 (WomenClothingEcommerceSalesData.csv)
    - Index set $I$: "Product Name"
    - Parameter $A_i$: "Revenue"
    - Parameter $d_i$: "Demand"
    - Parameter $s_i$: "Initial Inventory"