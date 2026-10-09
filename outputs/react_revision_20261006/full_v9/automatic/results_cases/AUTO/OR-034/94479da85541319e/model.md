#### Symbolic Mathematical Model

Let:

- $I$ = index set of all baked goods (from the current dataset)
- For each $i \in I$:
    - $A_i$ = revenue per unit of baked good $i$ (parameter)
    - $d_i$ = deterministic demand for baked good $i$ (parameter)
    - $s_i$ = initial inventory for baked good $i$ (parameter)
    - $x_i$ = quantity of baked good $i$ to fulfill (decision variable)

**Objective:**
\[
\max \sum_{i \in I} A_i \cdot x_i
\]

**Constraints:**
1. Inventory constraint:
\[
x_i \leq s_i \qquad \forall i \in I
\]

2. Demand constraint:
\[
x_i \leq d_i \qquad \forall i \in I
\]

3. Nonnegativity and integrality:
\[
x_i \in \mathbb{Z}_+, \qquad \forall i \in I
\]

#### Data Mapping

- Index set $I$: All rows in table_id = file_0_view_0, column "Product Name"
- Parameter $A_i$: table_id = file_0_view_0, column "Revenue"
- Parameter $d_i$: table_id = file_0_view_0, column "Demand"
- Parameter $s_i$: table_id = file_0_view_0, column "Initial Inventory"
- Decision variable $x_i$: quantity to fulfill for each $i \in I$