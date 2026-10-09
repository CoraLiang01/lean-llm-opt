#### Mathematical Model

Let:
- $\mathcal{I}$: Index set of all products classified under ‘Books’ (from Product_Name column).
- For each $i \in \mathcal{I}$:
    - $A_i$: Revenue per unit of product $i$ (from Revenue column).
    - $d_i$: Demand for product $i$ (from Demand column).
    - $I_i$: Initial Inventory for product $i$ (from Initial Inventory column).
    - $x_i$: Number of units of product $i$ to fulfill (decision variable).

Objective:
\[
\max \sum_{i \in \mathcal{I}} A_i \cdot x_i
\]

Subject to:
\[
\begin{align*}
& x_i \leq d_i, && \forall i \in \mathcal{I} \quad \text{(Demand constraint)} \\
& x_i \leq I_i, && \forall i \in \mathcal{I} \quad \text{(Inventory constraint)} \\
& x_i \geq 0, && \forall i \in \mathcal{I} \quad \text{(Non-negativity)} \\
& x_i \in \mathbb{Z}, && \forall i \in \mathcal{I} \quad \text{(Integer domain, if required)}
\end{align*}
\]

#### Data Mapping

- Table: file_0_view_0 (from DifferentStoreSales.csv)
    - Index set $\mathcal{I}$: All rows where Product_Name has prefix "Books"
    - $A_i$: Revenue (column "Revenue")
    - $d_i$: Demand (column "Demand")
    - $I_i$: Initial Inventory (column "Initial Inventory")