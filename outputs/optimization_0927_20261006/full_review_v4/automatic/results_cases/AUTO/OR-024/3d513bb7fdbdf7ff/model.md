Abstract Mathematical Model

Index Sets:
- Let $\mathcal{I}$ be the set of all products with identifiers starting with ‘S700_’.

Parameters:
- $A_i$: Revenue per unit of product $i \in \mathcal{I}$ (from column ‘Revenue’)
- $I_i$: Initial inventory of product $i \in \mathcal{I}$ (from column ‘Initial Inventory’)
- $d_i$: Demand for product $i \in \mathcal{I}$ (from column ‘Demand’)

Decision Variables:
- $x_i$: Number of units of product $i \in \mathcal{I}$ to fulfill, $x_i \in \mathbb{Z}_+$

Objective:
\[
\max \sum_{i \in \mathcal{I}} A_i \cdot x_i
\]

Constraints:
1. Inventory constraint for each product:
   \[
   x_i \leq I_i \quad \forall i \in \mathcal{I}
   \]
2. Demand constraint for each product:
   \[
   x_i \leq d_i \quad \forall i \in \mathcal{I}
   \]
3. Non-negativity and integrality:
   \[
   x_i \in \mathbb{Z}_+, \quad \forall i \in \mathcal{I}
   \]

Data Mapping:
- Table: SampleSalesData.csv, table_id: file_0_view_0
- Columns used:
    - Product Name (filtered: prefix = ‘S700_’) → index set $\mathcal{I}$
    - Revenue → parameter $A_i$
    - Initial Inventory → parameter $I_i$
    - Demand → parameter $d_i$