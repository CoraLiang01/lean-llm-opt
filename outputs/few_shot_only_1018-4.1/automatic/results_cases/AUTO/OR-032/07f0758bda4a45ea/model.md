#### Abstract Mathematical Optimization Model

**Index Sets:**
- $I$: Set of all products classified as 'Books'.

**Parameters:**
- $A_i$: Revenue per unit for product $i \in I$.  
  (From column 'Revenue' in /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM23/DifferentStoreSales.csv)
- $d_i$: Deterministic demand for product $i \in I$.  
  (From column 'Demand' in the same file)
- $I_i$: Initial inventory for product $i \in I$.  
  (From column 'Initial Inventory' in the same file)

**Decision Variables:**
- $x_i$: Number of units of product $i \in I$ to fulfill.  
  ($x_i \in \mathbb{Z}_+, \forall i \in I$)

**Objective:**
\[
\max \sum_{i \in I} A_i \cdot x_i
\]

**Constraints:**
1. **Demand fulfillment:**  
   \[
   x_i \leq d_i, \quad \forall i \in I
   \]
2. **Inventory availability:**  
   \[
   x_i \leq I_i, \quad \forall i \in I
   \]
3. **Non-negativity and integrality:**  
   \[
   x_i \in \mathbb{Z}_+, \quad \forall i \in I
   \]

---

#### Data Mapping

- **Source file:**  
  /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM23/DifferentStoreSales.csv

- **Index set $I$:**  
  All rows where the 'Product_Name' column value contains the substring 'Books'.

- **Parameter $A_i$:**  
  Value from column 'Revenue' for each $i \in I$.

- **Parameter $d_i$:**  
  Value from column 'Demand' for each $i \in I$.

- **Parameter $I_i$:**  
  Value from column 'Initial Inventory' for each $i \in I$.

- **Decision variable $x_i$:**  
  Defined for each $i \in I$.

No literal data values or record counts are included. All columns and rows are preserved as returned by the CSVQA tool.