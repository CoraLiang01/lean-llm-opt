### Abstract Mathematical Model

#### Index Sets
- $I$: Set of all products classified under ‘Baby’.

#### Parameters
- $A_i$: Revenue per unit of product $i \in I$ (from column ‘Revenue’ in table_id=file_0_view_0).
- $d_i$: Total demand for product $i \in I$ (from column ‘Demand’ in table_id=file_0_view_0).
- $I_i$: Initial inventory for product $i \in I$ (from column ‘Initial Inventory’ in table_id=file_0_view_0).

#### Decision Variables
- $x_i$: Number of units of product $i \in I$ to fulfill, $x_i \in \mathbb{Z}_+$.

#### Objective
\[
\max \sum_{i \in I} A_i \cdot x_i
\]

#### Constraints
1. Inventory constraint:
   \[
   x_i \leq I_i \quad \forall i \in I
   \]
2. Demand constraint:
   \[
   x_i \leq d_i \quad \forall i \in I
   \]
3. Non-negativity and integrality:
   \[
   x_i \in \mathbb{Z}_+, \quad \forall i \in I
   \]

---

#### Data Mapping

- Source Table: file_0_view_0 (EuropeSalesRecords.csv)
- Index Set $I$: All rows where ‘Product Name’ indicates classification under ‘Baby’ (as per user query and table structure).
- Parameter $A_i$: Column ‘Revenue’
- Parameter $d_i$: Column ‘Demand’
- Parameter $I_i$: Column ‘Initial Inventory’