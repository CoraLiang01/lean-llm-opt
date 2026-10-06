#### Abstract Mathematical Model

Let:
- $I$ = set of products, indexed by $i$ (with business identifier: "Product Name" from RetailSalesDataset.csv)
- For each $i \in I$:
    - $r_i$ = revenue per unit of product $i$ ("Revenue", file_0_view_0)
    - $d_i$ = demand for product $i$ ("Demand", file_0_view_0)
    - $s_i$ = initial inventory for product $i$ ("Initial Inventory", file_0_view_0)
- Decision variable: $x_i$ = quantity of product $i$ to fulfill (allocate to demand), $x_i \in \mathbb{Z}_{\geq 0}$

Objective:
$$
\max \sum_{i \in I} r_i x_i
$$

Subject to:
- Inventory limit for each product:
$$
x_i \leq s_i \quad \forall i \in I
$$
- Demand limit for each product:
$$
x_i \leq d_i \quad \forall i \in I
$$
- Nonnegativity and integrality:
$$
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
$$

#### Data Mapping

- $I$: All records in RetailSalesDataset.csv, file_0_view_0, column "Product Name"
- $r_i$: RetailSalesDataset.csv, file_0_view_0, column "Revenue", keyed by "Product Name"
- $d_i$: RetailSalesDataset.csv, file_0_view_0, column "Demand", keyed by "Product Name"
- $s_i$: RetailSalesDataset.csv, file_0_view_0, column "Initial Inventory", keyed by "Product Name"

Each constraint and parameter is mapped directly to the corresponding column and business identifier in the supplied data.