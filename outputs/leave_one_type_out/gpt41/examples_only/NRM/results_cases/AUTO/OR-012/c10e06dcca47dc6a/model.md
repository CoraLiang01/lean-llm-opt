ABSTRACT MATHEMATICAL OPTIMIZATION MODEL

Index Sets:
- 𝑰: Set of products, indexed by i.

Parameters:
- 𝑟ᵢ: Per-unit revenue for product i. [Source: OnlineSalesDataset.csv, column 'Revenue']
- 𝑑ᵢ: Deterministic demand for product i. [Source: OnlineSalesDataset.csv, column 'Demand']
- 𝑠ᵢ: Initial inventory for product i. [Source: OnlineSalesDataset.csv, column 'Initial Inventory']

Decision Variables:
- 𝑥ᵢ: Number of units of product i to fulfill for customer purchases (continuous or integer, as appropriate; domain: 0 ≤ 𝑥ᵢ ≤ min{𝑑ᵢ, 𝑠ᵢ}).

Objective:
- Maximize total revenue:
\[
\max \sum_{i \in 𝑰} rᵢ xᵢ
\]

Constraints:
1. Inventory constraint for each product:
\[
xᵢ \leq sᵢ \quad \forall i \in 𝑰
\]
2. Demand constraint for each product:
\[
xᵢ \leq dᵢ \quad \forall i \in 𝑰
\]
3. Non-negativity:
\[
xᵢ \geq 0 \quad \forall i \in 𝑰
\]

Data Mapping:
- Table: OnlineSalesDataset.csv (table_id: file_0_view_0)
    - Product index set 𝑰: 'Product Name'
    - Per-unit revenue 𝑟ᵢ: 'Revenue'
    - Initial inventory 𝑠ᵢ: 'Initial Inventory'
    - Demand 𝑑ᵢ: 'Demand'

This model maximizes total revenue by optimally allocating inventory to fulfill deterministic demand, subject to inventory and demand limits for each product.