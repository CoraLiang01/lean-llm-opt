#### Mathematical Optimization Model

Let:
- $I$ = index set of all products (from the "Product Name" column)
- For each $i \in I$:
    - $A_i$ = revenue per unit of product $i$ (from "Revenue")
    - $d_i$ = demand for product $i$ (from "Demand")
    - $s_i$ = initial inventory for product $i$ (from "Initial Inventory")
    - $x_i$ = number of units of product $i$ to fulfill (decision variable)

**Variables:**
- $x_i \in \mathbb{Z}_+, \quad \forall i \in I$

**Objective:**
\[
\max \sum_{i \in I} A_i x_i
\]

**Constraints:**
\[
\begin{align*}
& x_i \leq d_i, \quad \forall i \in I \\
& x_i \leq s_i, \quad \forall i \in I \\
& x_i \geq 0, \quad x_i \in \mathbb{Z}, \quad \forall i \in I
\end{align*}
\]

#### Data Mapping

- Table: file_0_view_0 (RetailSalesDataset.csv)
    - Index set $I$: All unique values in column "Product Name"
    - Parameter $A_i$: column "Revenue"
    - Parameter $d_i$: column "Demand"
    - Parameter $s_i$: column "Initial Inventory"
    - Decision variable $x_i$: fulfillment quantity for product $i$