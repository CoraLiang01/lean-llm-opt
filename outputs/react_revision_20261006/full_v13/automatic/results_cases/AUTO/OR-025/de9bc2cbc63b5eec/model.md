#### Symbolic Mathematical Model

Let:
- $I$ = index set of all products with ‘Product Name’ starting with "TABLET" in table_id file_0_view_0.
- For each $i \in I$:
    - $A_i$ = revenue per unit of product $i$ (parameter from column "Revenue")
    - $d_i$ = deterministic demand for product $i$ (parameter from column "Demand")
    - $s_i$ = initial inventory for product $i$ (parameter from column "Initial Inventory")
    - $x_i$ = number of units of product $i$ to fulfill (decision variable, integer, $x_i \geq 0$)

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

3. Nonnegativity and integrality:
$$
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
$$

#### Data Mapping

- Index set $I$: All rows in table_id file_0_view_0 where column "Product Name" starts with "TABLET" (from SmartphoneRetailOutletSalesData.csv).
- Parameter $A_i$: column "Revenue" in table_id file_0_view_0.
- Parameter $d_i$: column "Demand" in table_id file_0_view_0.
- Parameter $s_i$: column "Initial Inventory" in table_id file_0_view_0.
- Variable $x_i$: defined for each $i \in I$ as above.