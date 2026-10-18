#### Abstract Mathematical Model

**Index Set:**
- $I$ : Set of all products classified under ‘id999’ in /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM2/OnlineRetailSalesDataset.csv.

**Parameters:**
- $a_i$ : Revenue per unit for product $i \in I$ (from column ‘Revenue’).
- $d_i$ : Deterministic demand for product $i \in I$ during the sales horizon (from column ‘Demand’).
- $s_i$ : Initial inventory available for product $i \in I$ (from column ‘Initial Inventory’).

**Decision Variables:**
- $x_i$ : Number of units of product $i \in I$ to fulfill, $x_i \in \mathbb{Z}_+$ (non-negative integer).

**Objective:**
\[
\max \sum_{i \in I} a_i x_i
\]

**Constraints:**
1. **Inventory Constraint:**  
   \[
   x_i \leq s_i \qquad \forall i \in I
   \]
2. **Demand Constraint:**  
   \[
   x_i \leq d_i \qquad \forall i \in I
   \]
3. **Non-negativity and Integrality:**  
   \[
   x_i \in \mathbb{Z}_+, \qquad \forall i \in I
   \]

---

#### Data Mapping

- **Source file:** /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM2/OnlineRetailSalesDataset.csv
- **Index set $I$:** All rows where ‘id_number’ = ‘id999’
- **Parameter $a_i$:** Column ‘Revenue’
- **Parameter $d_i$:** Column ‘Demand’
- **Parameter $s_i$:** Column ‘Initial Inventory’