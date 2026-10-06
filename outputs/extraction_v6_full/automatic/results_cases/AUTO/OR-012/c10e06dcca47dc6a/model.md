#### Sets
- $I$: Set of all products, indexed by $i$.

#### Parameters
- $r_i$: Revenue per unit of product $i$.  
  [CSVQA_DATA: file_0_view_0, column: Revenue, index: Product Name]
- $d_i$: Total deterministic demand for product $i$ over the sales horizon.  
  [CSVQA_DATA: file_0_view_0, column: Demand, index: Product Name]
- $s_i$: Initial inventory available for product $i$.  
  [CSVQA_DATA: file_0_view_0, column: Initial Inventory, index: Product Name]

#### Decision Variables
- $x_i$: Number of units of product $i$ to fulfill for customer purchases.  
  Domain: $x_i \in \mathbb{Z}_+, \forall i \in I$

#### Objective
$$
\max \sum_{i \in I} r_i x_i
$$

#### Constraints

1. Inventory constraint (cannot sell more than available inventory):
   $$
   x_i \leq s_i, \quad \forall i \in I
   $$

2. Demand constraint (cannot fulfill more than demand):
   $$
   x_i \leq d_i, \quad \forall i \in I
   $$

3. Non-negativity and integrality:
   $$
   x_i \in \mathbb{Z}_+, \quad \forall i \in I
   $$

---

#### Data Mapping

- $I$ (product index set): All unique values in [Product Name] from table_id file_0_view_0 (OnlineSalesDataset.csv).
- $r_i$: [Revenue] column, indexed by [Product Name], from table_id file_0_view_0.
- $d_i$: [Demand] column, indexed by [Product Name], from table_id file_0_view_0.
- $s_i$: [Initial Inventory] column, indexed by [Product Name], from table_id file_0_view_0.