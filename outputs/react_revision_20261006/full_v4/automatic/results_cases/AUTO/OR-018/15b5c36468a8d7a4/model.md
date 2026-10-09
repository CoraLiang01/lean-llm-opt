##### Sets
Let $I$ be the set of all products classified under ‘Baby’ in the dataset.

##### Parameters
For each $i \in I$:
- $A_i$: Revenue per unit of product $i$ (from column ‘Revenue’)
- $d_i$: Demand for product $i$ (from column ‘Demand’)
- $I_i$: Initial inventory of product $i$ (from column ‘Initial Inventory’)

##### Decision Variables
For each $i \in I$:
- $x_i$: Number of units of product $i$ to fulfill, $x_i \in \mathbb{Z}_+$

##### Objective
\[
\max \sum_{i \in I} A_i \cdot x_i
\]

##### Constraints
\[
x_i \leq d_i \qquad \forall i \in I
\]
\[
x_i \leq I_i \qquad \forall i \in I
\]
\[
x_i \geq 0 \text{ and integer} \qquad \forall i \in I
\]

##### Data Mapping
- Table: Salesdata.csv (table_id: file_0_view_0)
    - Product Name: index set $I$ (filtered to ‘Baby’ products)
    - Revenue: parameter $A_i$
    - Demand: parameter $d_i$
    - Initial Inventory: parameter $I_i$