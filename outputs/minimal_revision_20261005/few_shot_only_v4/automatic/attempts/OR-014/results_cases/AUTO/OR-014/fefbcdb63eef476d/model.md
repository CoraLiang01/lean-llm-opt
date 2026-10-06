#### Abstract Mathematical Model

**Index Sets:**

- $I$ : set of all pizza types (indexed by $i$)

**Parameters:**

- $A_i$ : revenue per unit of pizza type $i$ (from column "Revenue")
- $d_i$ : total demand for pizza type $i$ over the sales horizon (from column "Demand")
- $I_i$ : initial inventory available for pizza type $i$ (from column "Initial Inventory")

**Decision Variables:**

- $x_i$ : number of units of pizza type $i$ to fulfill, $x_i \in \mathbb{Z}_+, \forall i \in I$

**Objective:**

$$
\max \sum_{i \in I} A_i \cdot x_i
$$

**Constraints:**

1. **Inventory Constraints:**
   $$
   x_i \leq I_i, \quad \forall i \in I
   $$
2. **Demand Constraints:**
   $$
   x_i \leq d_i, \quad \forall i \in I
   $$
3. **Non-negativity and Integrality:**
   $$
   x_i \in \mathbb{Z}_+, \quad \forall i \in I
   $$

---

#### Data Mapping

- **Table:** `/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM5/PizzaSalesDataset.csv`
    - **Index Set $I$:** `"Product Name"`
    - **Parameter $A_i$:** `"Revenue"`
    - **Parameter $d_i$:** `"Demand"`
    - **Parameter $I_i$:** `"Initial Inventory"`