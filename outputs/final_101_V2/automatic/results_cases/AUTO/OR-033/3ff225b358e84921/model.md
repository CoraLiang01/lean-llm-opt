Abstract Mathematical Model

Index Sets:
- $I$: Set of all products classified under ‘Baby’.

Parameters:
- $A_i$: Revenue per unit of product $i \in I$ (from column ‘Revenue’).
- $d_i$: Total demand for product $i \in I$ (from column ‘Demand’).
- $I_i$: Initial inventory for product $i \in I$ (from column ‘Initial Inventory’).

Decision Variables:
- $x_i$: Number of units of product $i \in I$ to fulfill, $x_i \in \mathbb{Z}_+, \forall i \in I$.

Objective:
\[
\max \sum_{i \in I} A_i \cdot x_i
\]

Constraints:
1. Inventory constraint:
   \[
   x_i \leq I_i, \quad \forall i \in I
   \]
2. Demand constraint:
   \[
   x_i \leq d_i, \quad \forall i \in I
   \]
3. Non-negativity and integrality:
   \[
   x_i \in \mathbb{Z}_+, \quad \forall i \in I
   \]

Data Mapping:
- Table: EuropeSalesRecords.csv (table_id: file_0_view_0)
    - Index set $I$: All rows where ‘Product Name’ has prefix ‘Baby’
    - Parameter $A_i$: column ‘Revenue’
    - Parameter $d_i$: column ‘Demand’
    - Parameter $I_i$: column ‘Initial Inventory’