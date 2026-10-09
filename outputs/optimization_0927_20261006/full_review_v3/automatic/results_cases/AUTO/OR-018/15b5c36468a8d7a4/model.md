Abstract Optimization Model

Index Sets:
- Let $I$ be the set of all products classified under ‘Baby’ (from column ‘Product Name’ with prefix "Baby" in table_id file_0_view_0).

Parameters:
- $A_i$: Revenue per unit of product $i \in I$ (from column ‘Revenue’).
- $d_i$: Total demand for product $i \in I$ (from column ‘Demand’).
- $I_i$: Initial inventory for product $i \in I$ (from column ‘Initial Inventory’).

Decision Variables:
- $x_i$: Number of units of product $i \in I$ to fulfill, $x_i \in \mathbb{Z}_+$.

Objective:
\[
\max \sum_{i \in I} A_i \cdot x_i
\]

Constraints:
1. Inventory and Demand Fulfillment:
\[
0 \leq x_i \leq \min\{I_i, d_i\} \quad \forall i \in I
\]

2. Integrality:
\[
x_i \in \mathbb{Z}_+ \quad \forall i \in I
\]

Data Mapping:
- Table: file_0_view_0 (from Salesdata.csv)
- Index Set $I$: All records where ‘Product Name’ has prefix "Baby"
- Parameter $A_i$: column ‘Revenue’
- Parameter $d_i$: column ‘Demand’
- Parameter $I_i$: column ‘Initial Inventory’