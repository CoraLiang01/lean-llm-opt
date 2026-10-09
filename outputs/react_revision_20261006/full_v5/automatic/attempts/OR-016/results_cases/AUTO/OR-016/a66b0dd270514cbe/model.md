#### Mathematical Optimization Model

Let:
- $\mathcal{I}$: Index set of all products (from Product Name in file_0_view_0)
- For each $i \in \mathcal{I}$:
    - $A_i$: Revenue per unit of product $i$ (from Revenue)
    - $d_i$: Demand for product $i$ (from Demand)
    - $I_i$: Initial Inventory for product $i$ (from Initial Inventory)
    - $x_i$: Number of units of product $i$ to fulfill (decision variable)

**Variables:**
- $x_i \in \mathbb{Z}_+, \quad \forall i \in \mathcal{I}$

**Objective:**
\[
\max \sum_{i \in \mathcal{I}} A_i \cdot x_i
\]

**Constraints:**
\[
\begin{align*}
& x_i \leq d_i, \quad \forall i \in \mathcal{I} \\
& x_i \leq I_i, \quad \forall i \in \mathcal{I} \\
& x_i \geq 0, \quad \forall i \in \mathcal{I}
\end{align*}
\]

#### Data Mapping

- Table: file_0_view_0 (RetailSalesDataset.csv)
    - Index set $\mathcal{I}$: Product Name
    - Parameter $A_i$: Revenue
    - Parameter $d_i$: Demand
    - Parameter $I_i$: Initial Inventory