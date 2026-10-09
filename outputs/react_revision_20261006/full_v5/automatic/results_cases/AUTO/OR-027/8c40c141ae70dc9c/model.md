#### Symbolic Mathematical Model

Let:
- $I$ = index set of all products with ‘Organ’ in the Sub Category (from the data)
- For each $i \in I$:
    - $A_i$ = revenue per unit of product $i$ (parameter, from Revenue column)
    - $d_i$ = demand for product $i$ (parameter, from Demand column)
    - $s_i$ = initial inventory of product $i$ (parameter, from Initial Inventory column)
    - $x_i$ = number of units of product $i$ to fulfill (decision variable, integer, $x_i \geq 0$)

Objective:
$$
\max \sum_{i \in I} A_i x_i
$$

Subject to:
- Inventory constraints:
$$
x_i \leq s_i \quad \forall i \in I
$$

- Demand constraints:
$$
x_i \leq d_i \quad \forall i \in I
$$

- Non-negativity and integrality:
$$
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
$$

#### Data Mapping

- Table: file_0_view_0 (SupermartGrocerySales-RetailAnalyticsDataset.csv)
    - Index set $I$: All rows where Sub Category has prefix "Organ"
    - Parameter $A_i$: column "Revenue"
    - Parameter $d_i$: column "Demand"
    - Parameter $s_i$: column "Initial Inventory"
    - Variable $x_i$: defined for each $i \in I$