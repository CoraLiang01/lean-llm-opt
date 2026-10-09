#### Index Sets

- $I$: set of all products (from column "Product Name" in table_id: file_0_view_0)

#### Parameters

- $r_i$: revenue per unit of product $i$ (from column "Revenue" in table_id: file_0_view_0)
- $d_i$: deterministic demand for product $i$ over the sales horizon (from column "Demand" in table_id: file_0_view_0)
- $s_i$: initial inventory of product $i$ (from column "Initial Inventory" in table_id: file_0_view_0)

#### Decision Variables

- $x_i$: number of units of product $i$ to fulfill for customer purchases, $\forall i \in I$; $x_i \in \mathbb{Z}_+$

#### Objective

$$
\max \sum_{i \in I} r_i \cdot x_i
$$

#### Constraints

1. Inventory and Demand Fulfillment:
   $$
   0 \leq x_i \leq \min\{d_i,\, s_i\}, \quad \forall i \in I
   $$

#### Data Mapping

- All sets and parameters are mapped from table_id: file_0_view_0, columns:
    - Product Name $\rightarrow$ $I$
    - Revenue $\rightarrow$ $r_i$
    - Demand $\rightarrow$ $d_i$
    - Initial Inventory $\rightarrow$ $s_i$