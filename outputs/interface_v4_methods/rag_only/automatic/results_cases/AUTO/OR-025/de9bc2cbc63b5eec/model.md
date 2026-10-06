ABSTRACT OPTIMIZATION MODEL

Index Sets:
- I: Set of all TABLET smartphone models, where each i ∈ I corresponds to a unique value in SmartphoneRetailOutletSalesData.csv, column 'Product Name' with prefix 'TABLET'.

Parameters:
- r_i: Revenue per unit for model i ∈ I. (Source: SmartphoneRetailOutletSalesData.csv, column 'Revenue')
- s_i: Initial inventory for model i ∈ I. (Source: SmartphoneRetailOutletSalesData.csv, column 'Initial Inventory')
- d_i: Demand for model i ∈ I. (Source: SmartphoneRetailOutletSalesData.csv, column 'Demand')

Decision Variables:
- x_i: Number of units of model i ∈ I to fulfill (integer, 0 ≤ x_i ≤ min{s_i, d_i})

Objective:
- Maximize total revenue:
  \[
  \max \sum_{i \in I} r_i x_i
  \]

Constraints:
1. Inventory constraint for each model:
   \[
   x_i \leq s_i \quad \forall i \in I
   \]
2. Demand constraint for each model:
   \[
   x_i \leq d_i \quad \forall i \in I
   \]
3. Non-negativity and integrality:
   \[
   x_i \geq 0,\ x_i \in \mathbb{Z} \quad \forall i \in I
   \]

Data Mapping:
- Index set I, parameters r_i, s_i, d_i, and all constraints are sourced from SmartphoneRetailOutletSalesData.csv (table_id: file_0_view_0):
    - 'Product Name' (with prefix 'TABLET') → I
    - 'Revenue' → r_i
    - 'Initial Inventory' → s_i
    - 'Demand' → d_i