#### Symbolic Optimization Model

Let:
- $I$ = index set of all ‘TABLET’ smartphone models (from all rows where ‘Product Name’ has prefix ‘TABLET’ in SmartphoneRetailOutletSalesData.csv)
- For each $i \in I$:
    - $A_i$ = revenue per unit of model $i$ (parameter from column ‘Revenue’)
    - $d_i$ = demand for model $i$ (parameter from column ‘Demand’)
    - $s_i$ = initial inventory for model $i$ (parameter from column ‘Initial Inventory’)
    - $x_i$ = number of units of model $i$ to fulfill (decision variable)

Objective:
$$
\max \sum_{i \in I} A_i \cdot x_i
$$

Subject to:
- Inventory and demand fulfillment bounds:
$$
0 \leq x_i \leq \min\{d_i,\, s_i\} \quad \forall i \in I
$$

- Variable domain:
$$
x_i \in \mathbb{Z} \quad \forall i \in I
$$

#### Data Mapping

- Table: SmartphoneRetailOutletSalesData.csv
    - Index set $I$: All rows where ‘Product Name’ has prefix ‘TABLET’ (column ‘Product Name’)
    - Parameter $A_i$: column ‘Revenue’
    - Parameter $d_i$: column ‘Demand’
    - Parameter $s_i$: column ‘Initial Inventory’