#### Index Sets
- $I$: Set of all products classified as ‘Fashion’ (from SupermarketSales.csv, see Data Mapping).

#### Parameters
- $r_i$: Revenue per unit of product $i \in I$ (SupermarketSales.csv, column ‘Revenue’).
- $d_i$: Demand for product $i \in I$ (SupermarketSales.csv, column ‘Demand’).
- $s_i$: Initial inventory for product $i \in I$ (SupermarketSales.csv, column ‘Initial Inventory’).

#### Decision Variables
- $x_i$: Number of units of product $i \in I$ to fulfill, integer, $x_i \geq 0$.

#### Objective
\[
\max \sum_{i \in I} r_i x_i
\]

#### Constraints
1. Inventory constraint:
   \[
   x_i \leq s_i \quad \forall i \in I
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

- Table: SupermarketSales.csv
- Table ID: file_0_view_0
- Filter: All rows where ‘Product Name’ has prefix ‘Fashion’ (i.e., products classified under ‘Fashion’)
- Columns used:
    - ‘Product Name’ (index set $I$)
    - ‘Revenue’ (parameter $r_i$)
    - ‘Demand’ (parameter $d_i$)
    - ‘Initial Inventory’ (parameter $s_i$)