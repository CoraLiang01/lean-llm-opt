#### Abstract Mathematical Model

**Index Set:**  
Let $I$ be the set of all products with "27in" in the "Product Name" column.

**Parameters:**  
For each $i \in I$:
- $A_i$: Revenue per unit of product $i$ (from column "Revenue")
- $d_i$: Demand for product $i$ (from column "Demand")
- $I_i$: Initial inventory of product $i$ (from column "Initial Inventory")

**Decision Variables:**  
For each $i \in I$:
- $x_i$: Number of units of product $i$ to fulfill  
  Domain: $x_i \in \mathbb{Z}_+, \quad 0 \leq x_i \leq \min\{d_i, I_i\}$

**Objective:**  
Maximize total revenue:
$$
\max \sum_{i \in I} A_i x_i
$$

**Constraints:**
1. **Demand fulfillment:**  
  $x_i \leq d_i \quad \forall i \in I$
2. **Inventory limit:**  
  $x_i \leq I_i \quad \forall i \in I$
3. **Non-negativity and integrality:**  
  $x_i \in \mathbb{Z}_+, \quad \forall i \in I$

---

#### Data Mapping

- **Source Table:**  
  /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM13/Salesorders.csv

- **Columns Used:**  
  "Product Name" (for index set $I$ and filtering "27in" products)  
  "Revenue" (parameter $A_i$)  
  "Demand" (parameter $d_i$)  
  "Initial Inventory" (parameter $I_i$)