#### Symbolic Model

**Index Sets:**
- $I$: set of all products (indexed by $i$), corresponding to all "Product Name" entries.

**Parameters:**
- $A_i$: revenue per unit of product $i$ (from column "Revenue").
- $d_i$: deterministic demand for product $i$ over the sales horizon (from column "Demand").
- $I_i$: initial inventory available for product $i$ (from column "Initial Inventory").

**Decision Variables:**
- $x_i$: number of units of product $i$ to fulfill for customer purchases, $\forall i \in I$.

**Objective:**
\[
\max \sum_{i \in I} A_i \cdot x_i
\]

**Constraints:**
1. **Inventory Constraint:**  
   \[
   x_i \leq I_i \quad \forall i \in I
   \]
2. **Demand Constraint:**  
   \[
   x_i \leq d_i \quad \forall i \in I
   \]
3. **Nonnegativity and Integrality:**  
   \[
   x_i \in \mathbb{Z}_+, \quad \forall i \in I
   \]

#### Data Mapping

- Table: `/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM3/OnlineSalesDataset.csv`
    - Product set $I$: column `"Product Name"`
    - Revenue parameter $A_i$: column `"Revenue"`
    - Demand parameter $d_i$: column `"Demand"`
    - Initial inventory parameter $I_i$: column `"Initial Inventory"`