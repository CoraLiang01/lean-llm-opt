##### Mathematical Model

Let:
- $I$ = set of products with Product_Reference starting with "ELE-S" (from file_0_view_0)
- For each $i \in I$:
    - $r_i$ = Revenue for product $i$ (column 'Revenue')
    - $d_i$ = Demand for product $i$ (column 'Demand')
    - $s_i$ = Initial Inventory for product $i$ (column 'Initial Inventory')
    - $x_i$ = number of units of product $i$ to fulfill (decision variable)

Objective:
$$
\max \sum_{i \in I} r_i x_i
$$

Subject to:
- Demand fulfillment (cannot exceed demand):
$$
x_i \leq d_i \quad \forall i \in I
$$

- Inventory availability (cannot exceed initial inventory):
$$
x_i \leq s_i \quad \forall i \in I
$$

- Nonnegativity and integrality:
$$
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
$$

##### Data Mapping

- Index set $I$: All records in file_0_view_0 (SalesStoreoverview.csv) with Product_Reference prefix "ELE-S"
- $r_i$: file_0_view_0, column 'Revenue', key 'Product_Reference'
- $d_i$: file_0_view_0, column 'Demand', key 'Product_Reference'
- $s_i$: file_0_view_0, column 'Initial Inventory', key 'Product_Reference'
- $x_i$: Decision variable for each $i \in I$ (Product_Reference in file_0_view_0)