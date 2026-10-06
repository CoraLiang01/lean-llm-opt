ABSTRACT MATHEMATICAL MODEL

Sets:
- $I$: Set of products classified under ‘Baby’, indexed by $i$ (from column ‘Product Name’ in table_id file_0_view_0).

Parameters (from table_id file_0_view_0):
- $r_i$: Revenue per unit of product $i$ (‘Revenue’)
- $d_i$: Demand for product $i$ (‘Demand’)
- $s_i$: Initial inventory of product $i$ (‘Initial Inventory’)

Decision Variables:
- $x_i$: Number of units of product $i$ to fulfill; $x_i \in \mathbb{Z}_{\geq 0}$

Objective:
\[
\max \sum_{i \in I} r_i x_i
\]

Subject to:
\[
x_i \leq d_i \qquad \forall i \in I \hspace{2cm} \text{(Demand constraint)}
\]
\[
x_i \leq s_i \qquad \forall i \in I \hspace{2cm} \text{(Inventory constraint)}
\]
\[
x_i \in \mathbb{Z}_{\geq 0} \qquad \forall i \in I
\]

Data Mapping:
- $I$: All rows in file_0_view_0 where ‘Product Name’ starts with ‘Baby’
- $r_i$: file_0_view_0, column ‘Revenue’, key ‘Product Name’
- $d_i$: file_0_view_0, column ‘Demand’, key ‘Product Name’
- $s_i$: file_0_view_0, column ‘Initial Inventory’, key ‘Product Name’