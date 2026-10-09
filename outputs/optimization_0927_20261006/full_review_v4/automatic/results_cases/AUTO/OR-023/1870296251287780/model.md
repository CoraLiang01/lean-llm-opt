### Abstract Mathematical Optimization Model

#### Index Sets
- $I$: Set of all products classified under ‘ELE-S’ (indexed by $i$).

#### Parameters
- $r_i$: Revenue per unit for product $i$ (from column ‘Revenue’).
- $d_i$: Deterministic demand for product $i$ (from column ‘Demand’).
- $s_i$: Initial inventory for product $i$ (from column ‘Initial Inventory’).

#### Decision Variables
- $x_i$: Number of units of product $i$ to fulfill, $\forall i \in I$.

#### Objective
\[
\max \sum_{i \in I} r_i \cdot x_i
\]

#### Constraints
1. Inventory and Demand Fulfillment:
   \[
   0 \leq x_i \leq \min\{d_i,\, s_i\}, \quad \forall i \in I
   \]
   (Each fulfilled quantity cannot exceed either available inventory or demand, and must be non-negative.)

2. Variable Domain:
   \[
   x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i \in I
   \]

---

### Data Mapping

- Table: SalesStoreoverview.csv (table_id: file_0_view_0)
    - Index set $I$: All records where ‘Product_Reference’ starts with ‘ELE-S’ (CSVQA filter: prefix = ‘ELE-S’).
    - Parameter $r_i$: Column ‘Revenue’.
    - Parameter $d_i$: Column ‘Demand’.
    - Parameter $s_i$: Column ‘Initial Inventory’.