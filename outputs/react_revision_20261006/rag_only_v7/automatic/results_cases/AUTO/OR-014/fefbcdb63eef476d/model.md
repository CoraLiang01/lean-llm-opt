Mathematical Optimization Model (Abstract Formulation)

Index Sets:
- 𝑃: Set of all pizza types, indexed by i. (from "Product Name" in table_id: file_0_view_0)

Parameters:
- r_i: Revenue per unit of pizza type i. (from "Revenue" in table_id: file_0_view_0)
- d_i: Demand for pizza type i. (from "Demand" in table_id: file_0_view_0)
- s_i: Initial inventory available for pizza type i. (from "Initial Inventory" in table_id: file_0_view_0)

Decision Variables:
- x_i: Number of units of pizza type i to fulfill, integer, x_i ≥ 0.

Objective:
- Maximize total revenue:
  \[
  \max_{x} \sum_{i \in P} r_i \cdot x_i
  \]

Constraints:
1. Inventory and demand fulfillment bounds for each pizza type:
   \[
   0 \leq x_i \leq \min\{d_i,\, s_i\} \quad \forall i \in P
   \]
   (Equivalently, two constraints per i:)
   \[
   x_i \leq d_i \quad \forall i \in P
   \]
   \[
   x_i \leq s_i \quad \forall i \in P
   \]
2. Integrality:
   \[
   x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in P
   \]

Data Mapping:
- Index set 𝑃: All unique values in "Product Name" (table_id: file_0_view_0)
- Parameter r_i: "Revenue" (table_id: file_0_view_0, column "Revenue")
- Parameter d_i: "Demand" (table_id: file_0_view_0, column "Demand")
- Parameter s_i: "Initial Inventory" (table_id: file_0_view_0, column "Initial Inventory")
- Decision variable x_i: Number of units fulfilled for pizza type i

No additional constraints or data sources are required per the user query.