#### Symbolic Mathematical Model

Let:
- $I$ = index set of all car models with Product Name prefix ‘FDK57’ (from the data)
- For each $i \in I$:
    - $A_i$ = revenue per unit of car model $i$ (parameter from column ‘Revenue’)
    - $d_i$ = demand for car model $i$ (parameter from column ‘Demand’)
    - $s_i$ = initial inventory for car model $i$ (parameter from column ‘Initial Inventory’)
    - $x_i$ = number of units of car model $i$ to fulfill (decision variable, integer, $0 \leq x_i \leq \min\{d_i, s_i\}$)

Objective:
$$
\max \sum_{i \in I} A_i x_i
$$

Subject to:
1. Demand and Inventory Fulfillment Bounds:
$$
0 \leq x_i \leq \min\{d_i, s_i\} \quad \forall i \in I
$$

2. Integrality:
$$
x_i \in \mathbb{Z} \quad \forall i \in I
$$

#### Data Mapping

- Table: file_0_view_0 (from BigMartSales.csv)
    - Index set $I$: all rows where [Product Name] has prefix ‘FDK57’
    - Parameter $A_i$: column [Revenue]
    - Parameter $d_i$: column [Demand]
    - Parameter $s_i$: column [Initial Inventory]
    - Decision variable $x_i$: quantity fulfilled for each $i \in I$