Abstract Mathematical Optimization Model

Index Sets:
- Let $I$ be the set of all products classified as ‘Baby’ in the dataset.

Parameters:
- $A_i$: Revenue per unit of product $i \in I$ (from column ‘Revenue’ in table_id file_0_view_0)
- $d_i$: Total demand for product $i \in I$ (from column ‘Demand’ in table_id file_0_view_0)
- $I_i$: Initial inventory for product $i \in I$ (from column ‘Initial Inventory’ in table_id file_0_view_0)

Decision Variables:
- $x_i$: Number of units of product $i \in I$ to fulfill, $x_i \in \mathbb{Z}_+$

Objective:
\[
\max \sum_{i \in I} A_i \cdot x_i
\]

Constraints:
1. Inventory constraint:
\[
x_i \leq I_i \quad \forall i \in I
\]
2. Demand constraint:
\[
x_i \leq d_i \quad \forall i \in I
\]
3. Non-negativity and integrality:
\[
x_i \in \mathbb{Z}_+, \quad \forall i \in I
\]

Data Mapping:
- Table: Salesdata.csv (table_id: file_0_view_0)
- Columns:
    - Product Name: used to identify set $I$ (‘Baby’ products)
    - Revenue: parameter $A_i$
    - Demand: parameter $d_i$
    - Initial Inventory: parameter $I_i$