#### Abstract Mathematical Optimization Model

**Index Sets:**
- $I$: Set of all products classified as ‘TABLET’ in `/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM16/SmartphoneRetailOutletSalesData.csv`.

**Parameters:**
- $A_i$: Revenue per unit of product $i \in I$.  
  (From column `Revenue`)
- $d_i$: Deterministic demand for product $i \in I$.  
  (From column `Demand`)
- $s_i$: Initial inventory for product $i \in I$.  
  (From column `Initial Inventory`)

**Decision Variables:**
- $x_i$: Number of units of product $i \in I$ to fulfill.  
  ($x_i \in \mathbb{Z}_+, \forall i \in I$)

**Objective:**
\[
\max \sum_{i \in I} A_i \cdot x_i
\]

**Constraints:**
1. **Inventory Bound:**  
   \[
   x_i \leq s_i, \quad \forall i \in I
   \]
2. **Demand Bound:**  
   \[
   x_i \leq d_i, \quad \forall i \in I
   \]
3. **Nonnegativity and Integrality:**  
   \[
   x_i \in \mathbb{Z}_+, \quad \forall i \in I
   \]

---

#### Data Mapping

- **Index Set $I$:**  
  All rows in `/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM16/SmartphoneRetailOutletSalesData.csv` where the value in the `Product Name` column begins with `TABLET_`.

- **Parameter $A_i$:**  
  Value in the `Revenue` column for product $i$ in the above file.

- **Parameter $d_i$:**  
  Value in the `Demand` column for product $i$ in the above file.

- **Parameter $s_i$:**  
  Value in the `Initial Inventory` column for product $i$ in the above file.

- **Decision Variable $x_i$:**  
  Number of units to fulfill for each $i \in I$.

---

**All data is mapped directly from the specified columns and file, with no filtering or aggregation beyond the ‘TABLET’ classification as described above.**