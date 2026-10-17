#### Abstract Mathematical Model

**Index Sets:**
- $I$ : Set of all products classified under ‘id999’ in the file `/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM2/OnlineRetailSalesDataset.csv`.

**Parameters:**
- $A_i$ : Revenue per unit for product $i \in I$. (From column ‘Revenue’)
- $d_i$ : Demand for product $i \in I$ during the sales horizon. (From column ‘Demand’)
- $I_i$ : Initial inventory for product $i \in I$. (From column ‘Initial Inventory’)

**Decision Variables:**
- $x_i$ : Number of units of product $i \in I$ to fulfill, $x_i \in \mathbb{Z}_+, \forall i \in I$.

**Objective:**
\[
\max \sum_{i \in I} A_i \cdot x_i
\]

**Constraints:**
1. **Inventory Constraints:**
   \[
   x_i \leq I_i, \quad \forall i \in I
   \]
2. **Demand Constraints:**
   \[
   x_i \leq d_i, \quad \forall i \in I
   \]
3. **Non-negativity and Integrality:**
   \[
   x_i \in \mathbb{Z}_+, \quad \forall i \in I
   \]

---

#### Data Mapping

- **Source file:** `/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM2/OnlineRetailSalesDataset.csv`
- **Index set $I$:** All rows where `id_number` = ‘id999’.
- **Parameter $A_i$:** Column `Revenue`.
- **Parameter $d_i$:** Column `Demand`.
- **Parameter $I_i$:** Column `Initial Inventory`.

All columns and rows are preserved as returned by CSVQA. No additional selection or filtering is performed beyond the explicit subset for ‘id999’ as specified in the query.