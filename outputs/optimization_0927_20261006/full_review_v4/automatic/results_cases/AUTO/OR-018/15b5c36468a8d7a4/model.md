Abstract Mathematical Model

Index Sets:
- Let $\mathcal{I}$ be the set of all products classified under ‘Baby’ (as returned by the data mapping).

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

Constraints:
1. Inventory constraint:
   \[
   x_i \leq I_i \quad \forall i \in \mathcal{I}
   \]
2. Demand constraint:
   \[
   x_i \leq d_i \quad \forall i \in \mathcal{I}
   \]
3. Non-negativity and integrality:
   \[
   x_i \in \mathbb{Z}_+, \quad \forall i \in \mathcal{I}
   \]

Data Mapping:
- Table: Salesdata.csv (table_id: file_0_view_0)
- Filter: Product Name prefix = ‘Baby’ (all products classified under ‘Baby’)
- Columns used:
    - Product Name (index set $\mathcal{I}$)
    - Revenue (parameter $A_i$)
    - Demand (parameter $d_i$)
    - Initial Inventory (parameter $I_i$)