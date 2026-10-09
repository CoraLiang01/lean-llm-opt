Abstract Mathematical Optimization Model

Index Set:
- $I$: Set of all products classified under ‘27in’ (from column “Product Name” in table_id file_0_view_0).

Parameters:
- $A_i$: Revenue per unit of product $i$ (from column “Revenue” in table_id file_0_view_0).
- $d_i$: Total demand for product $i$ (from column “Demand” in table_id file_0_view_0).
- $I_i$: Initial inventory for product $i$ (from column “Initial Inventory” in table_id file_0_view_0).

Decision Variables:
- $x_i$: Number of units of product $i$ to fulfill, $\forall i \in I$.

Objective:
\[
\max \sum_{i \in I} A_i \cdot x_i
\]

Constraints:
1. Inventory and Demand Fulfillment:
   \[
   0 \leq x_i \leq \min\{I_i, d_i\}, \quad \forall i \in I
   \]
   (Each fulfilled quantity cannot exceed available inventory or demand, and must be non-negative.)

2. Variable Domain:
   \[
   x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i \in I
   \]
   (Fulfilled quantities are non-negative integers.)

Data Mapping:
- Table: file_0_view_0 (from Salesorders.csv)
    - Product Name: Index set $I$
    - Revenue: Parameter $A_i$
    - Demand: Parameter $d_i$
    - Initial Inventory: Parameter $I_i$
- Filter: Only records where “Product Name” has prefix ‘27in’ (as applied in CSVQA).