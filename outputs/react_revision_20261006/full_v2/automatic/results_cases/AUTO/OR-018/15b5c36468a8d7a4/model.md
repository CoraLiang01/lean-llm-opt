Symbolic Mathematical Model

Sets:
- $I$: Index set of all products classified under ‘Baby’ (from table_id file_0_view_0, column "Product Name").

Parameters:
- $A_i$: Revenue per unit of product $i$ (from table_id file_0_view_0, column "Revenue").
- $d_i$: Demand for product $i$ (from table_id file_0_view_0, column "Demand").
- $I_i$: Initial inventory for product $i$ (from table_id file_0_view_0, column "Initial Inventory").

Decision Variables:
- $x_i$: Number of units of product $i$ to fulfill, $\forall i \in I$.

Objective:
\[
\max \sum_{i \in I} A_i x_i
\]

Subject to:
\[
x_i \leq d_i \quad \forall i \in I
\]
\[
x_i \leq I_i \quad \forall i \in I
\]
\[
x_i \geq 0 \quad \forall i \in I
\]

Data Mapping:
- Index set $I$, parameters $A_i$, $d_i$, $I_i$ are defined by all records in table_id file_0_view_0, columns "Product Name", "Revenue", "Demand", "Initial Inventory" filtered for products classified under ‘Baby’.