ABSTRACT OPTIMIZATION MODEL

Index Sets:
- I: Set of all TABLET smartphone models, indexed by i. (From SmartphoneRetailOutletSalesData.csv, table_id: file_0_view_0, column: Product Name)

Parameters:
- r_i: Revenue per unit for model i. (table_id: file_0_view_0, column: Revenue)
- s_i: Initial inventory for model i. (table_id: file_0_view_0, column: Initial Inventory)
- d_i: Demand for model i. (table_id: file_0_view_0, column: Demand)

Decision Variables:
- x_i: Number of units of TABLET model i to fulfill (integer, 0 ≤ x_i ≤ min{s_i, d_i})

Objective:
- Maximize total revenue:
  \[
  \max \sum_{i \in I} r_i x_i
  \]

Constraints:
1. Inventory and demand fulfillment:
   \[
   0 \leq x_i \leq \min\{s_i, d_i\} \quad \forall i \in I
   \]

Data Mapping:
- Index set I and all parameters (r_i, s_i, d_i) are sourced from SmartphoneRetailOutletSalesData.csv (table_id: file_0_view_0), using:
    - Product Name (TABLET models only) → i ∈ I
    - Revenue → r_i
    - Initial Inventory → s_i
    - Demand → d_i

No additional constraints are imposed by the query. All variable bounds are unconditional.