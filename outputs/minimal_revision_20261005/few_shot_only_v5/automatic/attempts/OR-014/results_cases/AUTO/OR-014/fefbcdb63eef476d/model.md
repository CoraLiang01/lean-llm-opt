#### Abstract Mathematical Model

**Index Sets:**

- $I$ : set of all pizza types (from column "Product Name" in table PizzaSalesDataset.csv)

**Parameters:**

- $A_i$ : revenue per unit of pizza type $i$ (from column "Revenue")
- $d_i$ : total demand for pizza type $i$ (from column "Demand")
- $I_i$ : initial inventory for pizza type $i$ (from column "Initial Inventory")

**Decision Variables:**

- $x_i$ : number of units of pizza type $i$ to fulfill, $\forall i \in I$  
  $x_i \in \mathbb{Z}_+, \quad \forall i \in I$

**Objective:**

$$
\max \sum_{i \in I} A_i \cdot x_i
$$

**Constraints:**

1. **Inventory Constraints:**  
   $x_i \leq I_i, \quad \forall i \in I$

2. **Demand Constraints:**  
   $x_i \leq d_i, \quad \forall i \in I$

3. **Non-negativity and Integrality:**  
   $x_i \in \mathbb{Z}_+, \quad \forall i \in I$

---

#### Data Mapping

- **Table:** `/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM5/PizzaSalesDataset.csv`
    - **Index Set $I$:** column `"Product Name"`
    - **Parameter $A_i$:** column `"Revenue"`
    - **Parameter $d_i$:** column `"Demand"`
    - **Parameter $I_i$:** column `"Initial Inventory"`