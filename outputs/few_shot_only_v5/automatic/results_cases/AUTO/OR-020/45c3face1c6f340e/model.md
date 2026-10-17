#### Abstract Mathematical Model

**Index Sets:**
- $I$: Set of all products, indexed by $i$.

**Parameters:**
- $A_i$: Revenue per unit of product $i$.  
  (From column ‘Revenue’ in /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM11/SalesDatainBusinesses.csv)
- $d_i$: Deterministic demand for product $i$.  
  (From column ‘Demand’ in the same file)
- $I_i$: Initial inventory available for product $i$.  
  (From column ‘Initial Inventory’ in the same file)

**Decision Variables:**
- $x_i$: Number of units of product $i$ to fulfill, $\forall i \in I$.

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

- **Source file:**  
  /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM11/SalesDatainBusinesses.csv

- **Parameter mapping:**  
  - $A_i$: ‘Revenue’ column  
  - $d_i$: ‘Demand’ column  
  - $I_i$: ‘Initial Inventory’ column  
  - $I$: All rows in the file, using ‘Product Name’ as the product identifier

- **No subset or filter is applied; all rows and columns are included as per the protocol.**