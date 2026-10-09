Abstract Mathematical Optimization Model

Index Sets:
- Let $\mathcal{I}$ be the set of all products classified under ‘27in’.

Parameters:
- $A_i$: Revenue per unit of product $i \in \mathcal{I}$ (from column ‘Revenue’).
- $d_i$: Total demand for product $i \in \mathcal{I}$ (from column ‘Demand’).
- $I_i$: Initial inventory for product $i \in \mathcal{I}$ (from column ‘Initial Inventory’).

Decision Variables:
- $x_i$: Number of units of product $i \in \mathcal{I}$ to fulfill, where $x_i \in \mathbb{Z}_+$ (non-negative integers).

Objective:
\[
\max \sum_{i \in \mathcal{I}} A_i \cdot x_i
\]

Subject to:
\[
\begin{align*}
& x_i \leq I_i, \quad \forall i \in \mathcal{I} \quad \text{(Inventory constraint)} \\
& x_i \leq d_i, \quad \forall i \in \mathcal{I} \quad \text{(Demand constraint)} \\
& x_i \geq 0, \quad x_i \in \mathbb{Z}, \quad \forall i \in \mathcal{I} \quad \text{(Non-negativity and integrality)}
\end{align*}
\]

Data Mapping:
- Table: SalesDataAnalysis.csv (table_id: file_0_view_0)
    - Index set $\mathcal{I}$: All records where ‘Product Name’ has prefix ‘27in’
    - Parameter $A_i$: column ‘Revenue’
    - Parameter $d_i$: column ‘Demand’
    - Parameter $I_i$: column ‘Initial Inventory’