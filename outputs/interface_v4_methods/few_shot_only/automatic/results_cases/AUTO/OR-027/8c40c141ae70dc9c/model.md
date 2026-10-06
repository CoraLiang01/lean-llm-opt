#### Abstract Mathematical Model

Let:

- $I$ = index set of all products classified as ‘Organ’
- For each $i \in I$:
    - $A_i$ = revenue per unit of product $i$ (parameter, from column ‘Revenue’)
    - $d_i$ = total demand for product $i$ (parameter, from column ‘Demand’)
    - $s_i$ = initial inventory for product $i$ (parameter, from column ‘Initial Inventory’)
    - $x_i$ = number of units of product $i$ to fulfill (decision variable, integer, $0 \leq x_i \leq \min\{d_i, s_i\}$)

Objective:
$$
\max \sum_{i \in I} A_i \cdot x_i
$$

Subject to:
- Demand and inventory fulfillment constraints:
$$
0 \leq x_i \leq \min\{d_i, s_i\} \qquad \forall i \in I
$$
- Integrality:
$$
x_i \in \mathbb{Z} \qquad \forall i \in I
$$

#### Data Mapping

- Table: file_0_view_0 (from SupermartGrocerySales-RetailAnalyticsDataset.csv)
    - Index set $I$: all rows where ‘Sub Category’ is classified as ‘Organ’ (as defined by the user or data dictionary)
    - Parameter $A_i$: column ‘Revenue’
    - Parameter $d_i$: column ‘Demand’
    - Parameter $s_i$: column ‘Initial Inventory’