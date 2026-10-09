Symbolic Mathematical Model

Index Sets:
Let $I$ be the set of all products, where each $i \in I$ corresponds to a unique product from the data.

Parameters:
For each $i \in I$:
- $A_i$: Revenue per unit of product $i$ (from column "Revenue")
- $d_i$: Demand for product $i$ (from column "Demand")
- $s_i$: Initial inventory of product $i$ (from column "Initial Inventory")

Decision Variables:
For each $i \in I$:
- $x_i$: Number of units of product $i$ to fulfill, $x_i \in \mathbb{Z}_+$ (non-negative integers)

Objective:
\[
\max \sum_{i \in I} A_i x_i
\]

Constraints:
\[
\begin{align*}
& x_i \leq d_i, && \forall i \in I \quad \text{(Demand constraint)} \\
& x_i \leq s_i, && \forall i \in I \quad \text{(Inventory constraint)} \\
& x_i \geq 0, && \forall i \in I \quad \text{(Non-negativity and integrality)}
\end{align*}
\]

Data Mapping:
- Table: file_0_view_0 (RetailSalesDataset.csv)
    - Index set $I$: All unique values in column "Product Name"
    - Parameter $A_i$: Column "Revenue"
    - Parameter $d_i$: Column "Demand"
    - Parameter $s_i$: Column "Initial Inventory"
    - Decision variable $x_i$: Fulfillment quantity for each $i \in I$