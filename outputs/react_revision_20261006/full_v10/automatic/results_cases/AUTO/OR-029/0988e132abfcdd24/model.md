Mathematical Optimization Model

Index Sets:
- Let $I$ be the set of all products with names beginning with "FAUX" in table_id file_0_view_0, column "Product Name".

Parameters:
- $A_i$: Revenue per unit of product $i \in I$ (from column "Revenue").
- $d_i$: Demand for product $i \in I$ (from column "Demand").
- $s_i$: Initial inventory for product $i \in I$ (from column "Initial Inventory").

Decision Variables:
- $x_i$: Number of units of product $i \in I$ to fulfill, $x_i \in \mathbb{Z}_+$ (non-negative integers).

Objective:
\[
\max \sum_{i \in I} A_i x_i
\]

Constraints:
1. Inventory constraint:
\[
x_i \leq s_i \quad \forall i \in I
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
- Index set $I$, parameters $A_i$, $d_i$, $s_i$ are defined by all rows in table_id file_0_view_0 (from ZARASales.csv) where "Product Name" starts with "FAUX", using columns:
    - "Product Name" (for $i$)
    - "Revenue" (for $A_i$)
    - "Demand" (for $d_i$)
    - "Initial Inventory" (for $s_i$)