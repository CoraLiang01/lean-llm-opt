#### Abstract Mathematical Model

**Index Set:**  
Let $I$ be the set of all products with Product_Reference codes beginning with "ELE-S" (i.e., all 'ELE-S' products) from the table.

**Parameters:**  
For each $i \in I$:
- $a_i$: Revenue per unit of product $i$ (from column "Revenue")
- $d_i$: Demand for product $i$ (from column "Demand")
- $s_i$: Initial inventory of product $i$ (from column "Initial Inventory")

**Decision Variables:**  
For each $i \in I$:
- $x_i$: Number of units of product $i$ to fulfill  
  Domain: $x_i \in \mathbb{Z}_+$ (non-negative integers)

**Objective:**  
Maximize total revenue:
$$
\max \sum_{i \in I} a_i x_i
$$

**Constraints:**
1. **Inventory constraint:**  
   $$
   x_i \leq s_i \qquad \forall i \in I
   $$
2. **Demand constraint:**  
   $$
   x_i \leq d_i \qquad \forall i \in I
   $$
3. **Non-negativity and integrality:**  
   $$
   x_i \in \mathbb{Z}_+, \qquad \forall i \in I
   $$

---

#### Data Mapping

- **Index Set $I$:**  
  All records in `/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM14/SalesStoreoverview.csv` where `Product_Reference` starts with "ELE-S".

- **Parameter $a_i$:**  
  `/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM14/SalesStoreoverview.csv`, column `Revenue`

- **Parameter $d_i$:**  
  `/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM14/SalesStoreoverview.csv`, column `Demand`

- **Parameter $s_i$:**  
  `/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM14/SalesStoreoverview.csv`, column `Initial Inventory`