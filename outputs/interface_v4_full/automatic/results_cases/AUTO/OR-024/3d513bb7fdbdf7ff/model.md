#### Abstract Mathematical Model

Let:
- $I$ = index set of all products with identifier prefix ‘S700_’ (from column ‘Product Name’)
- For each $i \in I$:
    - $A_i$ = revenue per unit of product $i$ (from column ‘Revenue’)
    - $d_i$ = total demand for product $i$ (from column ‘Demand’)
    - $s_i$ = initial inventory of product $i$ (from column ‘Initial Inventory’)
    - $x_i$ = number of units of product $i$ to fulfill (decision variable)

Objective:
$$
\max \sum_{i \in I} A_i \cdot x_i
$$

Subject to:
1. Inventory constraints:
$$
x_i \leq s_i \quad \forall i \in I
$$

2. Demand constraints:
$$
x_i \leq d_i \quad \forall i \in I
$$

3. Non-negativity and integrality:
$$
x_i \in \mathbb{Z}_+, \quad \forall i \in I
$$

#### Data Mapping

- Table: SampleSalesData.csv
    - Index set $I$: rows where [Product Name] has prefix ‘S700_’
    - Parameter $A_i$: [Revenue]
    - Parameter $d_i$: [Demand]
    - Parameter $s_i$: [Initial Inventory]