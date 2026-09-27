Abstract Mathematical Model

Index Sets:
- Let $\mathcal{I}$ be the set of all products classified as ‘Books’ (indexed by $i$).

Parameters:
- $A_i$: Revenue per unit of product $i$ (“Revenue” column).
- $I_i$: Initial inventory of product $i$ (“Initial Inventory” column).
- $d_i$: Deterministic demand for product $i$ (“Demand” column).

Decision Variables:
- $x_i$: Number of units of product $i$ to fulfill, $x_i \in \mathbb{Z}_+$ (non-negative integers), for all $i \in \mathcal{I}$.

Objective:
\[
\max \sum_{i \in \mathcal{I}} A_i \cdot x_i
\]

Subject to:
1. Inventory constraints:
\[
x_i \leq I_i \quad \forall i \in \mathcal{I}
\]
2. Demand constraints:
\[
x_i \leq d_i \quad \forall i \in \mathcal{I}
\]
3. Non-negativity and integrality:
\[
x_i \in \mathbb{Z}_+, \quad \forall i \in \mathcal{I}
\]

Data Mapping

- Table: DifferentStoreSales.csv (table_id: file_0_view_0)
    - Index set $\mathcal{I}$: All rows where “Product_Name” has prefix ‘Books’
    - Parameter $A_i$: “Revenue” column
    - Parameter $I_i$: “Initial Inventory” column
    - Parameter $d_i$: “Demand” column