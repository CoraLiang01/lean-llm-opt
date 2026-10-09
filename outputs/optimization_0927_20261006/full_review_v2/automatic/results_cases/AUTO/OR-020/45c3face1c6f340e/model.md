#### Index Sets

- $I$: Set of all products (indexed by $i$), corresponding to all "Product Name" entries in table_id file_0_view_0.

#### Parameters

- $A_i$: Revenue per unit of product $i$ (from column "Revenue" in file_0_view_0).
- $d_i$: Demand for product $i$ (from column "Demand" in file_0_view_0).
- $I_i$: Initial inventory for product $i$ (from column "Initial Inventory" in file_0_view_0).

#### Decision Variables

- $x_i$: Number of units of product $i$ to fulfill, for all $i \in I$.

#### Objective

$$
\max \sum_{i \in I} A_i \cdot x_i
$$

#### Constraints

1. Inventory and Demand Bounds:
   $$
   0 \leq x_i \leq \min\{d_i, I_i\}, \quad \forall i \in I
   $$
   (Each fulfilled quantity cannot exceed available inventory or demand, and must be non-negative.)

2. Integrality:
   $$
   x_i \in \mathbb{Z}, \quad \forall i \in I
   $$

#### Data Mapping

- All sets and parameters are mapped from table_id file_0_view_0, columns:
    - "Product Name" $\rightarrow$ $I$
    - "Revenue" $\rightarrow$ $A_i$
    - "Demand" $\rightarrow$ $d_i$
    - "Initial Inventory" $\rightarrow$ $I_i$
- Data source: SalesDatainBusinesses.csv

No additional filters were applied; all records and columns were used as returned by CSVQA.