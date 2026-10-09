Abstract Optimization Model

Index Sets:
- I: Set of products classified under ‘ZZ’ (indexed by i), where each i corresponds to a SKU in the dataset.

Parameters:
- r_i: Revenue per unit for product i ∈ I. (from column [Revenue])
- s_i: Initial inventory for product i ∈ I. (from column [Initial Inventory])
- d_i: Demand for product i ∈ I. (from column [Demand])

Decision Variables:
- x_i: Number of units of product i ∈ I to fulfill (integer, 0 ≤ x_i ≤ min{s_i, d_i})

Objective:
- Maximize total revenue from fulfilled units:
  \[
  \max \sum_{i \in I} r_i \cdot x_i
  \]

Constraints:
1. Inventory and demand fulfillment bounds for each product:
   \[
   0 \leq x_i \leq \min\{s_i, d_i\} \quad \forall i \in I
   \]
   (i.e., cannot fulfill more than available inventory or demand)

Variable Domains:
- x_i ∈ ℤ_+ (non-negative integers), ∀ i ∈ I

Data Mapping:
- Index set I: All rows in RetailStoreSalesTransactions(ScannerData).csv where [SKU] has prefix 'ZZ' (table_id: file_0_view_0, column: SKU)
- Parameter r_i: RetailStoreSalesTransactions(ScannerData).csv, column [Revenue], table_id: file_0_view_0
- Parameter s_i: RetailStoreSalesTransactions(ScannerData).csv, column [Initial Inventory], table_id: file_0_view_0
- Parameter d_i: RetailStoreSalesTransactions(ScannerData).csv, column [Demand], table_id: file_0_view_0

No additional relationships or constraints are imposed beyond those specified.