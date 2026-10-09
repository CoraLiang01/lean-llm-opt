Abstract Mathematical Model

Index Sets:
- Let $\mathcal{I}$ be the set of all products classified under ‘Organ’ (as defined by the returned records).

Parameters:
- $A_i$: Revenue per unit of product $i \in \mathcal{I}$ (from column ‘Revenue’).
- $I_i$: Initial inventory of product $i \in \mathcal{I}$ (from column ‘Initial Inventory’).
- $d_i$: Demand for product $i \in \mathcal{I}$ (from column ‘Demand’).

Decision Variables:
- $x_i$: Number of units of product $i \in \mathcal{I}$ to fulfill, $x_i \in \mathbb{Z}_+$.

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
- Table: SupermartGrocerySales-RetailAnalyticsDataset.csv (table_id: file_0_view_0)
- Filter: Rows where ‘Sub Category’ has prefix ‘Organic’ (i.e., products classified under ‘Organ’)
- Columns used: ‘Revenue’ $\rightarrow A_i$, ‘Initial Inventory’ $\rightarrow I_i$, ‘Demand’ $\rightarrow d_i$