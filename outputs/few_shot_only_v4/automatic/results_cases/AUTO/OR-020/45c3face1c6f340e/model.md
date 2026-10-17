#### Abstract Mathematical Optimization Model

**Index Sets:**
- $I$: Set of all products, indexed by $i$.

**Parameters:**
- $r_i$: Revenue per unit of product $i$.  
  (From column ‘Revenue’ in /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM11/SalesDatainBusinesses.csv)
- $d_i$: Deterministic demand for product $i$.  
  (From column ‘Demand’ in the same file)
- $s_i$: Initial inventory available for product $i$.  
  (From column ‘Initial Inventory’ in the same file)

**Decision Variables:**
- $x_i$: Number of units of product $i$ to fulfill, $x_i \in \mathbb{Z}_+$ (non-negative integers), for all $i \in I$.

**Objective:**
\[
\max \sum_{i \in I} r_i \cdot x_i
\]

**Constraints:**
1. **Demand fulfillment cannot exceed demand:**
   \[
   x_i \leq d_i \quad \forall i \in I
   \]
2. **Demand fulfillment cannot exceed available inventory:**
   \[
   x_i \leq s_i \quad \forall i \in I
   \]
3. **Non-negativity and integrality:**
   \[
   x_i \in \mathbb{Z}_+, \quad \forall i \in I
   \]

---

**Data Mapping:**

- File: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM11/SalesDatainBusinesses.csv
    - Index set $I$: All rows in column ‘Product Name’
    - Parameter $r_i$: Column ‘Revenue’
    - Parameter $d_i$: Column ‘Demand’
    - Parameter $s_i$: Column ‘Initial Inventory’

No additional constraints or eligibility rules are specified in the query; all products in the file are included in $I$. The model uses all rows and columns as returned by the data source.