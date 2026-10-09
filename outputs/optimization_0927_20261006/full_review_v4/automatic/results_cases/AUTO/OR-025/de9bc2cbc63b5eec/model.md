#### Index Sets

- $I$: Set of all ‘TABLET’ smartphone models, indexed by $i$.

#### Parameters

- $r_i$: Revenue per unit for model $i$ (from column ‘Revenue’ in table_id file_0_view_0).
- $d_i$: Demand for model $i$ (from column ‘Demand’ in table_id file_0_view_0).
- $s_i$: Initial inventory for model $i$ (from column ‘Initial Inventory’ in table_id file_0_view_0).

#### Decision Variables

- $x_i$: Number of units of model $i$ to fulfill, $\forall i \in I$.

#### Objective

$$
\max \sum_{i \in I} r_i x_i
$$

#### Constraints

1. Inventory constraint:
   $$
   x_i \leq s_i, \quad \forall i \in I
   $$
2. Demand constraint:
   $$
   x_i \leq d_i, \quad \forall i \in I
   $$
3. Nonnegativity and integrality:
   $$
   x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i \in I
   $$

#### Data Mapping

- Table: SmartphoneRetailOutletSalesData.csv (table_id: file_0_view_0)
- Filter: Product Name starts with ‘TABLET_’
- Columns used:
    - Product Name (index set $I$)
    - Revenue (parameter $r_i$)
    - Demand (parameter $d_i$)
    - Initial Inventory (parameter $s_i$)