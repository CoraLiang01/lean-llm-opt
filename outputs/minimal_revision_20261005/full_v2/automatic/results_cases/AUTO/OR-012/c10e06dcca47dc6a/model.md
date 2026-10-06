#### Sets
- $I$: Set of all products (indexed by $i$).

#### Parameters
- $A_i$: Revenue per unit of product $i$.  
  [Data Mapping: table_id = file_0_view_0, column = Revenue]
- $d_i$: Total deterministic demand for product $i$ over the sales horizon.  
  [Data Mapping: table_id = file_0_view_0, column = Demand]
- $I_i$: Initial inventory available for product $i$.  
  [Data Mapping: table_id = file_0_view_0, column = Initial Inventory]

#### Decision Variables
- $x_i$: Number of units of product $i$ to fulfill for customer purchases.  
  Domain: $x_i \in \mathbb{Z}_+$ (non-negative integers), for all $i \in I$.

#### Objective
\[
\max \sum_{i \in I} A_i \cdot x_i
\]

#### Constraints

1. **Inventory Constraints**  
   \[
   x_i \leq I_i \qquad \forall i \in I
   \]

2. **Demand Constraints**  
   \[
   x_i \leq d_i \qquad \forall i \in I
   \]

3. **Non-negativity and Integrality**  
   \[
   x_i \in \mathbb{Z}_+, \qquad \forall i \in I
   \]

---

#### Data Mapping

- All parameters ($A_i$, $d_i$, $I_i$) are mapped from table_id = file_0_view_0 (file: OnlineSalesDataset.csv), using columns:
    - Revenue $\rightarrow A_i$
    - Demand $\rightarrow d_i$
    - Initial Inventory $\rightarrow I_i$
    - Product Name $\rightarrow$ index set $I$