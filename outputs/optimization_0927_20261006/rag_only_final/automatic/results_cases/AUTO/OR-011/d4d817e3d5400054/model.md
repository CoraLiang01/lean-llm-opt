Abstract Optimization Model

Index Sets:
- 𝑰: Set of products classified under ‘id999’ (from Products table, Product Category = 'id999', column Product ID).

Parameters:
- 𝑟ᵢ: Revenue per unit for product i ∈ 𝑰 (from Revenue column).
- 𝑑ᵢ: Demand for product i ∈ 𝑰 during the sales horizon (from Demand column).
- 𝑠ᵢ: Initial inventory available for product i ∈ 𝑰 (from Initial Inventory column).

Decision Variables:
- 𝑥ᵢ: Number of units of product i ∈ 𝑰 to fulfill; 𝑥ᵢ ∈ ℤ₊ (non-negative integers).

Objective:
Maximize total revenue:
\[
\max \sum_{i \in 𝑰} r_i x_i
\]

Constraints:
1. Inventory constraint for each product:
\[
x_i \leq s_i \quad \forall i \in 𝑰
\]
2. Demand constraint for each product:
\[
x_i \leq d_i \quad \forall i \in 𝑰
\]
3. Non-negativity and integrality:
\[
x_i \in \mathbb{Z}_+, \quad \forall i \in 𝑰
\]

Data Mapping:
- Table: OnlineRetailSalesDataset.csv (table_id: file_0_view_0)
    - Product ID: index set 𝑰
    - Revenue: parameter 𝑟ᵢ
    - Demand: parameter 𝑑ᵢ
    - Initial Inventory: parameter 𝑠ᵢ
    - Filter: Product Category = 'id999' (as per query; only these products are included)

This model maximizes total revenue from fulfilling demand for ‘id999’ products, subject to inventory and demand limits, with all data sources and mappings explicitly identified.