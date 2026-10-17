#### Abstract Mathematical Optimization Model

**Index Sets:**

- $I$: Set of all products classified under ‘Baby’.

**Parameters:**

- $A_i$: Revenue per unit of product $i \in I$.  
  (From column ‘Revenue’ in /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM24/EuropeSalesRecords.csv)
- $d_i$: Deterministic demand for product $i \in I$.  
  (From column ‘Demand’ in /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM24/EuropeSalesRecords.csv)
- $I_i$: Initial inventory for product $i \in I$.  
  (From column ‘Initial Inventory’ in /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM24/EuropeSalesRecords.csv)

**Decision Variables:**

- $x_i$: Number of units of product $i \in I$ to fulfill.  
  ($x_i \in \mathbb{Z}_+, \forall i \in I$)

**Objective Function:**

$$
\max \sum_{i \in I} A_i \cdot x_i
$$

**Constraints:**

1. **Inventory and Demand Bounds:**  
   $$
   0 \leq x_i \leq \min\{I_i, d_i\}, \quad \forall i \in I
   $$

2. **Integrality:**  
   $$
   x_i \in \mathbb{Z}_+, \quad \forall i \in I
   $$

---

#### Data Mapping

- **File:** /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM24/EuropeSalesRecords.csv
- **Columns Used:**
  - ‘Product Name’: Used to identify products classified under ‘Baby’ (i.e., $I$).
  - ‘Revenue’: Parameter $A_i$.
  - ‘Demand’: Parameter $d_i$.
  - ‘Initial Inventory’: Parameter $I_i$.

**Subset Selection:**  
The index set $I$ includes all rows where ‘Product Name’ is classified under ‘Baby’ (i.e., contains or equals ‘Baby’ as per the query). No other rows are included. All columns are used as described above.