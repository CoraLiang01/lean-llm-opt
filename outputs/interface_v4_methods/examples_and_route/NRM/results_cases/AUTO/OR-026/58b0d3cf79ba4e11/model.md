#### Abstract Mathematical Model

Let:
- $I$ = set of all products with 'Product Name' starting with "Fashion" (see Data Mapping).
- For each $i \in I$:
    - $r_i$ = revenue per unit of product $i$ ('Revenue')
    - $d_i$ = demand for product $i$ ('Demand')
    - $s_i$ = initial inventory of product $i$ ('Initial Inventory')
    - $x_i$ = number of units of product $i$ to fulfill (decision variable)

**Decision Variables:**
- $x_i \in \mathbb{Z}_{\geq 0}$, for all $i \in I$

**Objective:**
$$
\max \sum_{i \in I} r_i x_i
$$

**Constraints:**
1. Inventory and demand fulfillment:
   $$
   0 \leq x_i \leq \min\{d_i, s_i\}, \quad \forall i \in I
   $$

---

#### Data Mapping

- $I$ (products): All rows in SupermarketSales.csv where 'Product Name' starts with "Fashion" (see table_id: file_0_view_0, column: 'Product Name')
- $r_i$: 'Revenue' column (file_0_view_0)
- $d_i$: 'Demand' column (file_0_view_0)
- $s_i$: 'Initial Inventory' column (file_0_view_0)
- $x_i$: Decision variable for each $i \in I$

Each $i$ is identified by its 'Product Name' value in file_0_view_0. All parameters are mapped directly from the corresponding columns in the returned data.