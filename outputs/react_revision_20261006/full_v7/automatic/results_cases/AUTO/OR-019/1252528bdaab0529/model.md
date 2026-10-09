#### Mathematical Optimization Model

Let:
- $I$ = set of all products with "27in" in their Product Name from the data.
- For each $i \in I$:
    - $A_i$ = revenue per unit of product $i$ (parameter, from column "Revenue")
    - $d_i$ = demand for product $i$ (parameter, from column "Demand")
    - $s_i$ = initial inventory for product $i$ (parameter, from column "Initial Inventory")
    - $x_i$ = number of units of product $i$ to fulfill (decision variable, integer, $x_i \geq 0$)

Objective:
$$
\max \sum_{i \in I} A_i \cdot x_i
$$

Subject to:
- Demand and inventory bounds:
$$
0 \leq x_i \leq \min\{d_i,\, s_i\} \quad \forall i \in I
$$
- $x_i \in \mathbb{Z}$ (integer), $\forall i \in I$

#### Data Mapping

- Table: file_0_view_0 (from SalesDataAnalysis.csv)
    - Index set $I$: All rows where "Product Name" contains "27in"
    - Parameter $A_i$: column "Revenue"
    - Parameter $d_i$: column "Demand"
    - Parameter $s_i$: column "Initial Inventory"
    - Decision variable $x_i$: number of units fulfilled for each $i \in I$