#### Index Sets

- $I$: Set of all products in the selected categories (electronics, apparel, homeware, etc.), indexed by $i$.

#### Parameters

- $A_i$: Revenue per unit for product $i$ (from column "Revenue" in table_id file_0_view_0).
- $d_i$: Demand for product $i$ (from column "Demand" in table_id file_0_view_0).
- $I_i$: Initial inventory for product $i$ (from column "Initial Inventory" in table_id file_0_view_0).

#### Decision Variables

- $x_i$: Number of units of product $i$ to fulfill, $x_i \in \mathbb{Z}_+$ (non-negative integers), for all $i \in I$.

#### Objective

$$
\max \sum_{i \in I} A_i \cdot x_i
$$

#### Constraints

1. Inventory constraints:
   $$
   x_i \leq I_i \quad \forall i \in I
   $$
2. Demand constraints:
   $$
   x_i \leq d_i \quad \forall i \in I
   $$
3. Non-negativity and integrality:
   $$
   x_i \in \mathbb{Z}_+, \quad \forall i \in I
   $$

#### Data Mapping

- Source: table_id file_0_view_0 (RetailSalesDataset.csv)
    - "Product Name": Used to identify products in electronics, apparel, homeware categories (filtered by prefix: "Electronics", "Apparel", "Homeware").
    - "Revenue": Parameter $A_i$.
    - "Demand": Parameter $d_i$.
    - "Initial Inventory": Parameter $I_i$.