#### Abstract Mathematical Optimization Model

**Index Sets:**
- $I$: Set of all products classified as ‘Baby’.

**Parameters:**
- $A_i$: Revenue per unit of product $i \in I$.  
  (From column ‘Revenue’ in /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM9/Salesdata.csv)
- $d_i$: Total demand for product $i \in I$ over the sales horizon.  
  (From column ‘Demand’ in the same file)
- $I_i$: Initial inventory of product $i \in I$.  
  (From column ‘Initial Inventory’ in the same file)

**Decision Variables:**
- $x_i$: Number of units of product $i \in I$ to fulfill (integer, $x_i \geq 0$).

**Objective:**
\[
\max \sum_{i \in I} A_i \cdot x_i
\]

**Constraints:**
1. **Inventory Constraint:**  
   \[
   x_i \leq I_i, \quad \forall i \in I
   \]
2. **Demand Constraint:**  
   \[
   x_i \leq d_i, \quad \forall i \in I
   \]
3. **Non-negativity and Integrality:**  
   \[
   x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i \in I
   \]

---

#### Data Mapping

- **File:** /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM9/Salesdata.csv
- **Index Set $I$:** All rows where ‘Product Name’ indicates classification as ‘Baby’ (e.g., contains ‘Baby’).
- **Parameter $A_i$:** Column ‘Revenue’.
- **Parameter $d_i$:** Column ‘Demand’.
- **Parameter $I_i$:** Column ‘Initial Inventory’.