#### Abstract Mathematical Model

Let:

- $I$ = index set of all dairy products (from column "Full_Product_Name" in table_id file_0_view_0)
- For each $i \in I$:
    - $A_i$ = revenue per unit of product $i$ (from column "Revenue")
    - $d_i$ = deterministic demand for product $i$ (from column "Demand")
    - $s_i$ = initial inventory for product $i$ (from column "Initial Inventory")
    - $x_i$ = integer variable: number of units of product $i$ to fulfill

Objective:
$$
\max \sum_{i \in I} A_i \cdot x_i
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

- Variable domain:
$$
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
$$

#### Data Mapping

- Table: file_0_view_0 (DairyGoodsSalesDataset.csv)
    - Index set $I$: column "Full_Product_Name"
    - Parameter $A_i$: column "Revenue"
    - Parameter $d_i$: column "Demand"
    - Parameter $s_i$: column "Initial Inventory"
    - Variable $x_i$: defined for each $i \in I$