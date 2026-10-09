#### Symbolic Mathematical Model

Let:
- $I$ = index set of all baked goods (from the current data)
- For each $i \in I$:
    - $A_i$ = revenue per unit of baked good $i$ (parameter)
    - $d_i$ = demand for baked good $i$ (parameter)
    - $s_i$ = initial inventory of baked good $i$ (parameter)
    - $x_i$ = quantity of baked good $i$ to fulfill (decision variable)

**Variables:**
- $x_i \in \mathbb{Z}_+, \quad \forall i \in I$

**Objective:**
\[
\max \sum_{i \in I} A_i \cdot x_i
\]

**Constraints:**
\[
x_i \leq d_i, \quad \forall i \in I
\]
\[
x_i \leq s_i, \quad \forall i \in I
\]
\[
x_i \geq 0, \quad \forall i \in I
\]

#### Data Mapping

- Table: file_0_view_0 (Frenchbakerydailysales.csv)
    - Index set $I$: all unique values in column "Product Name"
    - Parameter $A_i$: column "Revenue"
    - Parameter $d_i$: column "Demand"
    - Parameter $s_i$: column "Initial Inventory"