#### Index Sets

- Let $\mathcal{I}$ be the set of all products with Sub Category prefix "Organic" (from column "Sub Category" in table_id file_0_view_0).

#### Parameters

- $A_i$: Revenue per unit of product $i$ (from column "Revenue", table_id file_0_view_0).
- $d_i$: Demand for product $i$ (from column "Demand", table_id file_0_view_0).
- $I_i$: Initial Inventory for product $i$ (from column "Initial Inventory", table_id file_0_view_0).

#### Decision Variables

- $x_i$: Number of units of product $i$ to fulfill, for all $i \in \mathcal{I}$.

#### Objective

$$
\max \sum_{i \in \mathcal{I}} A_i \cdot x_i
$$

#### Constraints

1. Inventory and Demand Bounds:
   $$
   0 \leq x_i \leq \min\{I_i,\, d_i\} \quad \forall i \in \mathcal{I}
   $$

2. Variable Domain:
   $$
   x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in \mathcal{I}
   $$

#### Data Mapping

- Table: file_0_view_0 (from SupermartGrocerySales-RetailAnalyticsDataset.csv)
    - Index set $\mathcal{I}$: All rows where "Sub Category" has prefix "Organic"
    - Parameter $A_i$: "Revenue"
    - Parameter $d_i$: "Demand"
    - Parameter $I_i$: "Initial Inventory"