#### Symbolic Mathematical Model

**Index Sets:**
- $I$: set of all products (indexed by $i$), where each $i$ corresponds to a unique "Product Name" in the dataset.

**Parameters:**
- $A_i$: revenue per unit of product $i$ (from column "Revenue").
- $d_i$: total demand for product $i$ (from column "Demand").
- $I_i$: initial inventory for product $i$ (from column "Initial Inventory").

**Decision Variables:**
- $x_i$: number of units of product $i$ to fulfill, $x_i \in \mathbb{Z}_+$ (non-negative integers), for all $i \in I$.

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
3. **Non-negativity and Integrality:**
   \[
   x_i \in \mathbb{Z}_+, \quad \forall i \in I
   \]

---

#### Data Mapping

- **Index Set $I$:** All unique values in column "Product Name" of table_id `/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM12/Salesofsummerclothes.csv`
- **Parameter $A_i$:** Column "Revenue" of the same table, mapped to each $i \in I$
- **Parameter $d_i$:** Column "Demand" of the same table, mapped to each $i \in I$
- **Parameter $I_i$:** Column "Initial Inventory" of the same table, mapped to each $i \in I$
- **Decision Variable $x_i$:** Defined for each $i \in I$ as above

All data is sourced from `/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM12/Salesofsummerclothes.csv` using columns "Product Name", "Revenue", "Demand", and "Initial Inventory".