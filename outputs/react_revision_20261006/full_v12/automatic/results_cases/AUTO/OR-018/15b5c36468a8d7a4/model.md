#### Symbolic Mathematical Model

Let:

- $I$ = index set of all products classified under ‘Baby’ (from column "Product Name" with prefix "Baby")
- For each $i \in I$:
    - $A_i$ = revenue per unit of product $i$ (from column "Revenue")
    - $d_i$ = demand for product $i$ (from column "Demand")
    - $s_i$ = initial inventory of product $i$ (from column "Initial Inventory")
    - $x_i$ = number of units of product $i$ to fulfill (decision variable)

Objective:
\[
\max \sum_{i \in I} A_i x_i
\]

Subject to:
\[
x_i \leq d_i \quad \forall i \in I
\]
\[
x_i \leq s_i \quad \forall i \in I
\]
\[
x_i \geq 0 \quad \forall i \in I
\]
\[
x_i \in \mathbb{Z} \quad \forall i \in I
\]

#### Data Mapping

- Table: file_0_view_0 (from Salesdata.csv)
    - Index set $I$: all rows where "Product Name" has prefix "Baby"
    - $A_i$: column "Revenue"
    - $d_i$: column "Demand"
    - $s_i$: column "Initial Inventory"