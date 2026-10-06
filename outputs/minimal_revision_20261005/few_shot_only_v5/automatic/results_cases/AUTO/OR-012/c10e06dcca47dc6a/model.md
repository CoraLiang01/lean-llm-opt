#### Abstract Mathematical Model

**Index Set:**  
Let $I$ be the set of all products in the dataset.

**Parameters:**  
For each $i \in I$:
- $A_i$: Revenue per unit of product $i$ (from column "Revenue")
- $d_i$: Total deterministic demand for product $i$ (from column "Demand")
- $I_i$: Initial inventory of product $i$ (from column "Initial Inventory")

**Decision Variables:**  
For each $i \in I$:
- $x_i$: Number of units of product $i$ to fulfill for customer purchases  
  Domain: $x_i \in \mathbb{Z}_+$ (non-negative integers)

**Objective:**  
Maximize total revenue:
$$
\max \sum_{i \in I} A_i \cdot x_i
$$

**Constraints:**
1. **Inventory Constraint:**  
  For all $i \in I$,
$$
x_i \leq I_i
$$

2. **Demand Constraint:**  
  For all $i \in I$,
$$
x_i \leq d_i
$$

3. **Non-negativity and Integrality:**  
  For all $i \in I$,
$$
x_i \in \mathbb{Z}_+, \quad x_i \geq 0
$$

---

#### Data Mapping

- **Source Table:**  
  `/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM3/OnlineSalesDataset.csv`
- **Columns Used:**  
  - "Product Name" (for index set $I$)  
  - "Revenue" (for parameter $A_i$)  
  - "Demand" (for parameter $d_i$)  
  - "Initial Inventory" (for parameter $I_i$)