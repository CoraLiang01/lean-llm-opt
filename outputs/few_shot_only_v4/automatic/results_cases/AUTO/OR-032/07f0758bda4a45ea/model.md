#### Abstract Mathematical Optimization Model

**Index Sets:**
- $I$: Set of all products classified as 'Books'.

**Parameters:**
- $A_i$: Revenue per unit for product $i \in I$.  
  (From column ‘Revenue’ in /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM23/DifferentStoreSales.csv)
- $d_i$: Deterministic demand for product $i \in I$.  
  (From column ‘Demand’ in the same file)
- $I_i$: Initial inventory for product $i \in I$.  
  (From column ‘Initial Inventory’ in the same file)

**Decision Variables:**
- $x_i$: Number of units of product $i \in I$ to fulfill.  
  ($x_i \in \mathbb{Z}_+, \forall i \in I$)

**Objective:**
\[
\max \sum_{i \in I} A_i \cdot x_i
\]

**Constraints:**
1. **Demand fulfillment constraint:**  
   \[
   x_i \leq d_i, \quad \forall i \in I
   \]
2. **Inventory constraint:**  
   \[
   x_i \leq I_i, \quad \forall i \in I
   \]
3. **Non-negativity and integrality:**  
   \[
   x_i \in \mathbb{Z}_+, \quad \forall i \in I
   \]

---

#### Data Mapping

- **File:** /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM23/DifferentStoreSales.csv
- **Columns Used:**
  - Product_Name (for identifying set $I$ as all rows where Product_Name indicates 'Books')
  - Revenue (parameter $A_i$)
  - Demand (parameter $d_i$)
  - Initial Inventory (parameter $I_i$)

No literal data values or record counts are included. All rows and columns are considered as returned by CSVQA; the set $I$ is defined by all products classified as 'Books' according to the Product_Name column.