#### Sets
Let $I$ be the set of all products with SKU prefix ‘ZZ’:
$$
I = \{\text{all SKUs in RetailStoreSalesTransactions(ScannerData).csv with prefix 'ZZ'}\}
$$

#### Parameters
For each $i \in I$:
- $r_i$: Revenue per unit of product $i$ (from column "Revenue")
- $d_i$: Demand for product $i$ (from column "Demand")
- $s_i$: Initial inventory of product $i$ (from column "Initial Inventory")

#### Decision Variables
For each $i \in I$:
- $x_i$: Number of units of product $i$ to fulfill, $x_i \in \mathbb{Z}_+$

#### Objective
$$
\max \sum_{i \in I} r_i x_i
$$

#### Constraints
- Demand and inventory limits:
$$
0 \leq x_i \leq \min\{d_i,\, s_i\} \quad \forall i \in I
$$

#### Data Mapping
- Table: RetailStoreSalesTransactions(ScannerData).csv
    - Index set $I$: All rows where column "SKU" has prefix 'ZZ'
    - Parameter $r_i$: column "Revenue"
    - Parameter $d_i$: column "Demand"
    - Parameter $s_i$: column "Initial Inventory"