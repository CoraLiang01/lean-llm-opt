##### Symbolic Mathematical Model

Let:
- $I$ = set of all products with "Product Name" starting with "27in" (from the data)
- For each $i \in I$:
    - $A_i$ = revenue per unit of product $i$ (parameter from column "Revenue")
    - $d_i$ = demand for product $i$ (parameter from column "Demand")
    - $s_i$ = initial inventory for product $i$ (parameter from column "Initial Inventory")
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

##### Data Mapping

- Table: file_0_view_0 (from SalesDataAnalysis.csv)
    - Index set $I$: All rows where "Product Name" starts with "27in"
    - Parameter $A_i$: column "Revenue"
    - Parameter $d_i$: column "Demand"
    - Parameter $s_i$: column "Initial Inventory"