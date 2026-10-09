#### Index Sets
- $I$: Set of all products classified under ‘Baby’ in Salesdata.csv (column ‘Product Name’).

#### Parameters
- $A_i$: Revenue per unit of product $i \in I$ (from column ‘Revenue’ in Salesdata.csv).
- $d_i$: Total demand for product $i \in I$ (from column ‘Demand’ in Salesdata.csv).
- $I_i$: Initial inventory for product $i \in I$ (from column ‘Initial Inventory’ in Salesdata.csv).

#### Decision Variables
- $x_i$: Number of units of product $i \in I$ to fulfill, integer, $x_i \geq 0$.

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
   x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
   \]

---

#### Data Mapping

- Source table: Salesdata.csv
- Table ID: file_0_view_0
- Columns used:
    - ‘Product Name’ (filtered: prefix = ‘Baby’)
    - ‘Revenue’
    - ‘Demand’
    - ‘Initial Inventory’