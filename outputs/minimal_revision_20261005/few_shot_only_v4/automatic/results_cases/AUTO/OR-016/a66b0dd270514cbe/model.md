#### Abstract Mathematical Model

**Index Sets:**

- $I$ : Set of all product categories (indexed by $i$), where each $i$ corresponds to a unique "Product Name".

**Parameters:**

- $A_i$ : Revenue per unit of product $i$ (from column "Revenue").
- $d_i$ : Total demand for product $i$ (from column "Demand").
- $I_i$ : Initial inventory available for product $i$ (from column "Initial Inventory").

**Decision Variables:**

- $x_i$ : Number of units of product $i$ to fulfill (integer, $x_i \geq 0$).

**Objective:**

$$
\max \sum_{i \in I} A_i \cdot x_i
$$

**Constraints:**

1. **Inventory Constraints:**  
   $$
   x_i \leq I_i \quad \forall i \in I
   $$

2. **Demand Constraints:**  
   $$
   x_i \leq d_i \quad \forall i \in I
   $$

3. **Non-negativity and Integrality:**  
   $$
   x_i \in \mathbb{Z}_{+} \quad \forall i \in I
   $$

---

#### Data Mapping

- **Index Set $I$:**  
  All unique values in column "Product Name" of table `/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM7/RetailSalesDataset.csv`.

- **Parameter $A_i$:**  
  Column "Revenue" of table `/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM7/RetailSalesDataset.csv`.

- **Parameter $d_i$:**  
  Column "Demand" of table `/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM7/RetailSalesDataset.csv`.

- **Parameter $I_i$:**  
  Column "Initial Inventory" of table `/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM7/RetailSalesDataset.csv`.