#### Mathematical Model

Let  
- $I$ = set of all products in the current data with "Product Name" containing the substring "Fashion" (i.e., all 'Fashion' products).

Parameters (for each $i \in I$):  
- $A_i$ = revenue per unit of product $i$ (from column "Revenue")  
- $d_i$ = total demand for product $i$ (from column "Demand")  
- $s_i$ = initial inventory for product $i$ (from column "Initial Inventory")

Decision variables:  
- $x_i$ = number of units of product $i$ to fulfill, $x_i \in \mathbb{Z}_+$ (non-negative integers), for all $i \in I$

Objective:  
$$
\max \sum_{i \in I} A_i x_i
$$

Subject to:  
- Demand and inventory bounds:
  $$
  0 \leq x_i \leq \min\{d_i, s_i\}, \quad \forall i \in I
  $$

- Integer constraints:
  $$
  x_i \in \mathbb{Z}_+, \quad \forall i \in I
  $$

#### Data Mapping

- Table: `/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM17/SupermarketSales.csv`, table_id: `file_0_view_0`
    - Index set $I$: All records where `Product Name` contains the substring "Fashion"
    - Parameter $A_i$: column `Revenue`
    - Parameter $d_i$: column `Demand`
    - Parameter $s_i$: column `Initial Inventory`