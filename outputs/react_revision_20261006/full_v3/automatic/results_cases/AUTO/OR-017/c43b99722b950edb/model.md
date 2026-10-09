#### Sets
Let $I$ be the set of all products with SKU prefix ‘ZZ’:
$$
I = \{\text{ZZ2AO},\ \text{ZZDW7},\ \text{ZZM1A},\ \text{ZZNC5},\ \text{ZZX6K}\}
$$

#### Parameters
For each $i \in I$:
- $A_i$: Revenue per unit of product $i$ (from column "Revenue")
- $d_i$: Demand for product $i$ (from column "Demand")
- $s_i$: Initial inventory of product $i$ (from column "Initial Inventory")

#### Decision Variables
For each $i \in I$:
- $x_i$: Number of units of product $i$ to fulfill, $x_i \in \mathbb{Z}_+$

#### Objective
$$
\max \sum_{i \in I} A_i x_i
$$

#### Constraints
- Demand and inventory bounds:
  $$
  0 \leq x_i \leq \min\{d_i,\, s_i\} \quad \forall i \in I
  $$

#### Data Mapping
- Table: file_0_view_0 (RetailStoreSalesTransactions(ScannerData).csv)
    - Index set $I$: All rows where column "SKU" has prefix ‘ZZ’
    - Parameter $A_i$: column "Revenue"
    - Parameter $d_i$: column "Demand"
    - Parameter $s_i$: column "Initial Inventory"