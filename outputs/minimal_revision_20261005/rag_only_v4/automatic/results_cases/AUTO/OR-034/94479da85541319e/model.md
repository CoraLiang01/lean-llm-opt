Mathematical Optimization Model (Abstract Formulation)

Index Sets:
- 𝑃 : Set of all baked goods products, indexed by 𝑖.  
  (𝑃 = all "Product Name" entries in table_id file_0_view_0)

Parameters:
- 𝑟ᵢ : Revenue per unit of product 𝑖  
  (Source: "Revenue" column, table_id file_0_view_0)
- 𝑑ᵢ : Demand quantity for product 𝑖  
  (Source: "Demand" column, table_id file_0_view_0)
- 𝑠ᵢ : Initial inventory available for product 𝑖  
  (Source: "Initial Inventory" column, table_id file_0_view_0)

Decision Variables:
- 𝑥ᵢ : Quantity of product 𝑖 to fulfill (integer, 0 ≤ 𝑥ᵢ ≤ min{𝑑ᵢ, 𝑠ᵢ}), ∀𝑖 ∈ 𝑃

Objective:
- Maximize total revenue:
\[
\max \sum_{i \in P} r_i x_i
\]

Constraints:
1. Demand fulfillment and inventory limits:
\[
0 \leq x_i \leq \min\{d_i, s_i\} \quad \forall i \in P
\]
  (Each product’s fulfilled quantity cannot exceed its demand or available inventory.)

Variable Domains:
- 𝑥ᵢ ∈ ℤ₊ (non-negative integers), ∀𝑖 ∈ 𝑃

---

Data Mapping

- Index set 𝑃: All "Product Name" in table_id file_0_view_0
- Parameter 𝑟ᵢ: "Revenue" column, table_id file_0_view_0
- Parameter 𝑑ᵢ: "Demand" column, table_id file_0_view_0
- Parameter 𝑠ᵢ: "Initial Inventory" column, table_id file_0_view_0