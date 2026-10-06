#### Abstract Mathematical Model

Let:
- $I$ = set of products, indexed by $i$ (with business identifier: "Product Name" from MobileSalesDataset.csv)
- For each $i \in I$:
    - $r_i$ = revenue per unit of product $i$ ("Revenue", MobileSalesDataset.csv)
    - $d_i$ = demand for product $i$ ("Demand", MobileSalesDataset.csv)
    - $s_i$ = initial inventory for product $i$ ("Initial Inventory", MobileSalesDataset.csv)
- Decision variables: $x_i$ = number of units of product $i$ fulfilled, $x_i \in \mathbb{Z}_{\geq 0}$

Objective:
$$
\max \sum_{i \in I} r_i x_i
$$

Subject to:
$$
x_i \leq d_i \qquad \forall i \in I \\
x_i \leq s_i \qquad \forall i \in I \\
x_i \in \mathbb{Z}_{\geq 0} \qquad \forall i \in I
$$

#### Data Mapping

- $I$: "Product Name" (MobileSalesDataset.csv)
- $r_i$: "Revenue" (MobileSalesDataset.csv, table_id: file_0_view_0)
- $d_i$: "Demand" (MobileSalesDataset.csv, table_id: file_0_view_0)
- $s_i$: "Initial Inventory" (MobileSalesDataset.csv, table_id: file_0_view_0)