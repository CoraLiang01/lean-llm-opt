ABSTRACT OPTIMIZATION MODEL

Index Sets:
- 𝑰: Set of all pizza types (indexed by i)

Parameters:
- r_i: Revenue per unit of pizza type i (from PizzaSalesDataset.csv, column "Revenue", table_id: file_0_view_0)
- d_i: Demand for pizza type i (from PizzaSalesDataset.csv, column "Demand", table_id: file_0_view_0)
- s_i: Initial inventory available for pizza type i (from PizzaSalesDataset.csv, column "Initial Inventory", table_id: file_0_view_0)

Decision Variables:
- x_i: Number of units of pizza type i to fulfill (integer, x_i ≥ 0, ∀i ∈ 𝑰)

Objective:
- Maximize total revenue:
  \[
  \max \sum_{i \in 𝑰} r_i \cdot x_i
  \]

Constraints:
1. Inventory and demand fulfillment:
   \[
   0 \leq x_i \leq \min\{d_i, s_i\} \quad \forall i \in 𝑰
   \]
   (Equivalently, two constraints per i:)
   \[
   x_i \leq d_i \quad \forall i \in 𝑰
   \]
   \[
   x_i \leq s_i \quad \forall i \in 𝑰
   \]
2. Integrality:
   \[
   x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in 𝑰
   \]

Data Mapping:
- Index set 𝑰: All unique values in "Product Name" from PizzaSalesDataset.csv (table_id: file_0_view_0)
- Parameter r_i: "Revenue" column from PizzaSalesDataset.csv (table_id: file_0_view_0)
- Parameter d_i: "Demand" column from PizzaSalesDataset.csv (table_id: file_0_view_0)
- Parameter s_i: "Initial Inventory" column from PizzaSalesDataset.csv (table_id: file_0_view_0)

No data values or record counts are included; all data is referenced symbolically by table and column.