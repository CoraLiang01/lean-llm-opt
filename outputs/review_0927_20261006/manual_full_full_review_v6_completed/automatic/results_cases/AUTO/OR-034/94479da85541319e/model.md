#### Index Sets
- $I$: set of all baked goods (from column "Product Name" in table_id: file_0_view_0).

#### Parameters
- $A_i$: revenue per unit of baked good $i$ (from column "Revenue" in table_id: file_0_view_0).
- $d_i$: deterministic demand for baked good $i$ (from column "Demand" in table_id: file_0_view_0).
- $I_i$: initial inventory for baked good $i$ (from column "Initial Inventory" in table_id: file_0_view_0).

#### Decision Variables
- $x_i$: quantity of baked good $i$ to fulfill, for all $i \in I$.

#### Objective
$$
\max \sum_{i \in I} A_i \cdot x_i
$$

#### Constraints
1. Inventory and Demand Fulfillment:
   $$
   0 \leq x_i \leq \min\{I_i, d_i\}, \quad \forall i \in I
   $$
   (Each fulfilled quantity cannot exceed available inventory or demand, and must be non-negative.)

2. Variable Domain:
   $$
   x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i \in I
   $$

#### Data Mapping
- Table: file_0_view_0 (Frenchbakerydailysales.csv)
    - Index set $I$: column "Product Name"
    - Parameter $A_i$: column "Revenue"
    - Parameter $d_i$: column "Demand"
    - Parameter $I_i$: column "Initial Inventory"