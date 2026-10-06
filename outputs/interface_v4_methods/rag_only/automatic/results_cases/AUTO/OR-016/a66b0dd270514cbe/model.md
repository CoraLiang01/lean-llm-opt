ABSTRACT OPTIMIZATION MODEL

Index Sets:
- Let 𝑃 be the set of all products/categories, indexed by i.

Parameters:
- 𝑟ᵢ: Revenue per unit for product i. [RetailSalesDataset.csv, column: Revenue, table_id: file_0_view_0]
- 𝑑ᵢ: Demand quantity for product i. [RetailSalesDataset.csv, column: Demand, table_id: file_0_view_0]
- 𝑠ᵢ: Initial inventory for product i. [RetailSalesDataset.csv, column: Initial Inventory, table_id: file_0_view_0]

Decision Variables:
- 𝑥ᵢ: Quantity of product i to fulfill (allocate to demand), ∀i ∈ 𝑃.
 Domain: 0 ≤ 𝑥ᵢ ≤ min{𝑑ᵢ, 𝑠ᵢ}, 𝑥ᵢ continuous (or integer if required by business rules).

Objective:
- Maximize total revenue:
  max ∑_{i ∈ 𝑃} 𝑟ᵢ · 𝑥ᵢ

Constraints:
1. Inventory limit:  𝑥ᵢ ≤ 𝑠ᵢ  ∀i ∈ 𝑃
2. Demand limit:   𝑥ᵢ ≤ 𝑑ᵢ  ∀i ∈ 𝑃
3. Non-negativity:  𝑥ᵢ ≥ 0   ∀i ∈ 𝑃

Data Mapping:
- Index set 𝑃: All unique values in [RetailSalesDataset.csv, column: Product Name, table_id: file_0_view_0]
- Parameter 𝑟ᵢ: [RetailSalesDataset.csv, column: Revenue, table_id: file_0_view_0]
- Parameter 𝑑ᵢ: [RetailSalesDataset.csv, column: Demand, table_id: file_0_view_0]
- Parameter 𝑠ᵢ: [RetailSalesDataset.csv, column: Initial Inventory, table_id: file_0_view_0]