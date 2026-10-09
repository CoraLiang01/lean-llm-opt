#### Symbolic Mathematical Model

Let $I$ be the set of all baked goods in the dataset.

Parameters:
- $A_i$: Revenue per unit of baked good $i$, from column "Revenue".
- $d_i$: Demand for baked good $i$, from column "Demand".
- $I_i$: Initial inventory for baked good $i$, from column "Initial Inventory".

Decision variables:
- $x_i$: Quantity of baked good $i$ to fulfill, $\forall i \in I$.

Objective:
$$
\max \sum_{i \in I} A_i x_i
$$

Subject to:
- Inventory constraints: $x_i \leq I_i, \quad \forall i \in I$
- Demand constraints: $x_i \leq d_i, \quad \forall i \in I$
- Nonnegativity and integrality: $x_i \in \mathbb{Z}_+, \quad \forall i \in I$

#### Data Mapping

- $I$: All "Product Name" entries in table_id: file_0_view_0, column: "Product Name"
- $A_i$: file_0_view_0, column: "Revenue"
- $d_i$: file_0_view_0, column: "Demand"
- $I_i$: file_0_view_0, column: "Initial Inventory"