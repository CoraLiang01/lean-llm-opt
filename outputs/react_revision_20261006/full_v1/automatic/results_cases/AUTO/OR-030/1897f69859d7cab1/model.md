#### Symbolic Mathematical Model

Let:
- $\mathcal{I}$ = index set of all car models classified under ‘FDK57’ (from the data)
- For each $i \in \mathcal{I}$:
    - $A_i$ = revenue per unit of car model $i$ (parameter, from ‘Revenue’)
    - $d_i$ = demand for car model $i$ (parameter, from ‘Demand’)
    - $I_i$ = initial inventory of car model $i$ (parameter, from ‘Initial Inventory’)
    - $x_i$ = number of units of car model $i$ to fulfill (decision variable)

Objective:
$$
\max \sum_{i \in \mathcal{I}} A_i \cdot x_i
$$

Subject to:
1. Inventory constraints:
$$
x_i \leq I_i \quad \forall i \in \mathcal{I}
$$

2. Demand constraints:
$$
x_i \leq d_i \quad \forall i \in \mathcal{I}
$$

3. Non-negativity and integrality:
$$
x_i \in \mathbb{Z}_+, \quad \forall i \in \mathcal{I}
$$

#### Data Mapping

- Table: file_0_view_0 (from BigMartSales.csv)
    - Index set $\mathcal{I}$: All rows where Product Name has prefix ‘FDK57’
    - Parameter $A_i$: column ‘Revenue’
    - Parameter $d_i$: column ‘Demand’
    - Parameter $I_i$: column ‘Initial Inventory’