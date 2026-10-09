##### Mathematical Optimization Model

Let $I$ be the index set of all car models classified under ‘FDK57’ in the dataset.

Parameters:
- $A_i$: Revenue per unit for car model $i \in I$ (from column ‘Revenue’)
- $d_i$: Demand for car model $i \in I$ (from column ‘Demand’)
- $s_i$: Initial inventory for car model $i \in I$ (from column ‘Initial Inventory’)

Decision Variables:
- $x_i$: Number of units of car model $i \in I$ to fulfill (integer, $0 \leq x_i \leq \min\{d_i, s_i\}$)

Objective:
$$
\max \sum_{i \in I} A_i x_i
$$

Subject to:
1. Demand constraint: $x_i \leq d_i \quad \forall i \in I$
2. Inventory constraint: $x_i \leq s_i \quad \forall i \in I$
3. Non-negativity and integrality: $x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I$

##### Data Mapping

- Table: file_0_view_0 (from BigMartSales.csv)
    - Index set $I$: All rows where ‘Product Name’ has prefix ‘FDK57’
    - Parameter $A_i$: column ‘Revenue’
    - Parameter $d_i$: column ‘Demand’
    - Parameter $s_i$: column ‘Initial Inventory’