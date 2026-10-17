#### Index Sets

- $I$: Set of all products classified under ‘ELE-S’ (indexed by $i$).

#### Parameters

- $r_i$: Revenue per unit for product $i \in I$ (from column ‘Revenue’).
- $d_i$: Total demand for product $i \in I$ (from column ‘Demand’).
- $s_i$: Initial inventory for product $i \in I$ (from column ‘Initial Inventory’).

#### Decision Variables

- $x_i$: Number of units of product $i \in I$ to fulfill, $x_i \in \mathbb{Z}_+$.

#### Objective

\[
\max \sum_{i \in I} r_i x_i
\]

#### Constraints

1. Inventory constraint for each product:
   \[
   x_i \leq s_i \quad \forall i \in I
   \]
2. Demand constraint for each product:
   \[
   x_i \leq d_i \quad \forall i \in I
   \]
3. Non-negativity and integrality:
   \[
   x_i \in \mathbb{Z}_+, \quad \forall i \in I
   \]

---

#### Data Mapping

- Table: SalesStoreoverview.csv
- Table ID: file_0_view_0
- Columns used:
    - Product_Reference (filtered by prefix ‘ELE-S’)
    - Revenue
    - Demand
    - Initial Inventory

Returned records are those where Product_Reference starts with ‘ELE-S’, as validated and returned by CSVQA.