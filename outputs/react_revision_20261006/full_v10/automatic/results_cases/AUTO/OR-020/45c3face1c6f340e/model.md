Symbolic Mathematical Model

Index Sets:
Let $I$ be the set of all products, indexed by $i$.

Parameters:
For each $i \in I$:
- $A_i$: revenue per unit of product $i$ (from column "Revenue")
- $d_i$: total demand for product $i$ (from column "Demand")
- $I_i$: initial inventory of product $i$ (from column "Initial Inventory")

Decision Variables:
For each $i \in I$:
- $x_i$: number of units of product $i$ to fulfill, $x_i \in \mathbb{Z}_+$

Objective:
\[
\max \sum_{i \in I} A_i x_i
\]

Constraints:
\[
x_i \leq d_i \quad \forall i \in I
\]
\[
x_i \leq I_i \quad \forall i \in I
\]
\[
x_i \geq 0, \quad x_i \in \mathbb{Z} \quad \forall i \in I
\]

Data Mapping:
- Index set $I$ and all parameters $A_i$, $d_i$, $I_i$ are defined by all rows in table_id file_0_view_0, columns "Product Name", "Revenue", "Demand", and "Initial Inventory" from SalesDatainBusinesses.csv.