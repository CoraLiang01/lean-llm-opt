#### Abstract Mathematical Optimization Model

**Index Sets:**
- $I$: Set of all pizza types (indexed by $i$).

**Parameters:**
- $A_i$: Revenue per unit for pizza type $i$.
- $d_i$: Total demand for pizza type $i$ over the sales horizon.
- $I_i$: Initial inventory available for pizza type $i$.

**Decision Variables:**
- $x_i \in \mathbb{Z}_+$: Number of units of pizza type $i$ to fulfill (non-negative integer).

**Objective:**
\[
\max \sum_{i \in I} A_i \cdot x_i
\]

**Constraints:**
1. **Inventory Constraints:**
   \[
   x_i \leq I_i \quad \forall i \in I
   \]
2. **Demand Constraints:**
   \[
   x_i \leq d_i \quad \forall i \in I
   \]
3. **Non-negativity and Integrality:**
   \[
   x_i \in \mathbb{Z}_+, \quad \forall i \in I
   \]

---

#### Data Mapping

- **Index Set $I$ (pizza types):** All rows in `/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM5/PizzaSalesDataset.csv`, column `Product Name`.
- **Parameter $A_i$ (revenue per unit):** `/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM5/PizzaSalesDataset.csv`, column `Revenue`.
- **Parameter $d_i$ (demand):** `/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM5/PizzaSalesDataset.csv`, column `Demand`.
- **Parameter $I_i$ (initial inventory):** `/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM5/PizzaSalesDataset.csv`, column `Initial Inventory`.