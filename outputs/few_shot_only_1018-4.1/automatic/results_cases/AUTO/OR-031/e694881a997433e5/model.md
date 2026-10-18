#### Abstract Mathematical Optimization Model

**Index Sets:**

- $I$: Set of all dairy products, indexed by $i$.

**Parameters:**

- $A_i$: Revenue per unit for product $i$.  
  (From column "Revenue" in /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM22/DairyGoodsSalesDataset.csv)
- $d_i$: Deterministic demand for product $i$.  
  (From column "Demand" in the same file)
- $I_i$: Initial inventory for product $i$.  
  (From column "Initial Inventory" in the same file)

**Decision Variables:**

- $x_i$: Number of units of product $i$ to fulfill, integer, $x_i \geq 0$.

**Objective:**

$$
\max \sum_{i \in I} A_i \cdot x_i
$$

**Constraints:**

1. **Inventory Constraint:**  
   $$
   x_i \leq I_i \quad \forall i \in I
   $$

2. **Demand Constraint:**  
   $$
   x_i \leq d_i \quad \forall i \in I
   $$

3. **Non-negativity and Integrality:**  
   $$
   x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
   $$

---

#### Data Mapping

- **Source file:**  
  /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM22/DairyGoodsSalesDataset.csv

- **Index set $I$:**  
  All rows in column "Full_Product_Name".

- **Parameter $A_i$:**  
  Column "Revenue".

- **Parameter $d_i$:**  
  Column "Demand".

- **Parameter $I_i$:**  
  Column "Initial Inventory".

- **Decision variable $x_i$:**  
  Number of units fulfilled for each product $i \in I$.

---

This model maximizes total revenue from fulfilling orders for all dairy products, subject to initial inventory and demand limits, using all data and identifiers as provided in the source file and columns.