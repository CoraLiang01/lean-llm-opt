Mathematical Optimization Model

Index Sets:
Let $I$ be the set of all products, indexed by $i$.

Parameters:
For each $i \in I$:
- $A_i$: Revenue per unit of product $i$ (from column "Revenue")
- $d_i$: Demand for product $i$ (from column "Demand")
- $I_i$: Initial inventory of product $i$ (from column "Initial Inventory")

Decision Variables:
For each $i \in I$:
- $x_i$: Number of units of product $i$ to fulfill, integer, $x_i \geq 0$

Objective:
\[
\max \sum_{i \in I} A_i x_i
\]

Constraints:
\[
\begin{align*}
& x_i \leq d_i, \quad \forall i \in I \\
& x_i \leq I_i, \quad \forall i \in I \\
& x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i \in I
\end{align*}
\]

Data Mapping:
- Table: file_0_view_0 (WomenClothingEcommerceSalesData.csv)
    - Index set $I$: All unique values in column "Product Name"
    - Parameter $A_i$: Column "Revenue"
    - Parameter $d_i$: Column "Demand"
    - Parameter $I_i$: Column "Initial Inventory"