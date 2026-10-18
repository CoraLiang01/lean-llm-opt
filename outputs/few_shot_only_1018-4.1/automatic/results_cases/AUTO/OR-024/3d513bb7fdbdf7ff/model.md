#### Abstract Mathematical Optimization Model

**Index Set:**
- Let $\mathcal{I}$ be the set of all products $i$ such that the value in the column ‘Product Name’ begins with ‘S700_’ in the file `/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM15/SampleSalesData.csv`.

**Parameters:**
- $A_i$: Revenue per unit of product $i$ (from column ‘Revenue’).
- $I_i$: Initial inventory of product $i$ (from column ‘Initial Inventory’).
- $d_i$: Demand for product $i$ (from column ‘Demand’).

**Decision Variables:**
- $x_i \in \mathbb{Z}_+$: Number of units of product $i$ to fulfill, for each $i \in \mathcal{I}$.

**Objective:**
\[
\max \sum_{i \in \mathcal{I}} A_i \cdot x_i
\]

**Constraints:**
1. **Inventory Constraint:**  
   \[
   x_i \leq I_i \qquad \forall i \in \mathcal{I}
   \]
2. **Demand Constraint:**  
   \[
   x_i \leq d_i \qquad \forall i \in \mathcal{I}
   \]
3. **Non-negativity and Integrality:**  
   \[
   x_i \in \mathbb{Z}_+, \qquad \forall i \in \mathcal{I}
   \]

---

#### Data Mapping

- **Source file:** `/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM15/SampleSalesData.csv`
- **Index set $\mathcal{I}$:** All rows where the value in column ‘Product Name’ starts with ‘S700_’.
- **Parameter $A_i$:** Value in column ‘Revenue’ for product $i$.
- **Parameter $I_i$:** Value in column ‘Initial Inventory’ for product $i$.
- **Parameter $d_i$:** Value in column ‘Demand’ for product $i$.
- **Decision variable $x_i$:** Number of units to fulfill for product $i \in \mathcal{I}$.

No additional constraints or selection rules are imposed beyond those stated above.