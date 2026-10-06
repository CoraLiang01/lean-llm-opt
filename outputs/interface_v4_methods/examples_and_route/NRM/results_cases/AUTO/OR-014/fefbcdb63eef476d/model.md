#### Abstract Mathematical Model

Let:
- $I$ = set of pizza types, indexed by $i$ (from "Product Name" in PizzaSalesDataset.csv)
- For each $i \in I$:
    - $r_i$ = revenue per unit of pizza type $i$ ("Revenue", table_id: file_0_view_0)
    - $d_i$ = demand for pizza type $i$ ("Demand", table_id: file_0_view_0)
    - $s_i$ = initial inventory of pizza type $i$ ("Initial Inventory", table_id: file_0_view_0)
- Decision variables: $x_i$ = number of units of pizza type $i$ to fulfill, $x_i \in \mathbb{Z}_{\geq 0}$

Objective:
$$
\max \sum_{i \in I} r_i x_i
$$

Subject to:
$$
x_i \leq d_i \quad \forall i \in I \\
x_i \leq s_i \quad \forall i \in I \\
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
$$

#### Data Mapping

- $I$: All records in PizzaSalesDataset.csv, column "Product Name", table_id: file_0_view_0
- $r_i$: "Revenue", table_id: file_0_view_0, matched by "Product Name"
- $d_i$: "Demand", table_id: file_0_view_0, matched by "Product Name"
- $s_i$: "Initial Inventory", table_id: file_0_view_0, matched by "Product Name"