#### Abstract Mathematical Model

**Index Set:**  
Let $I$ be the set of all products with identifiers beginning with ‘S700_’ in the table.

**Parameters:**  
For each $i \in I$:
- $A_i$: Revenue per unit of product $i$ (from column ‘Revenue’)
- $d_i$: Demand for product $i$ (from column ‘Demand’)
- $I_i$: Initial inventory of product $i$ (from column ‘Initial Inventory’)

**Decision Variables:**  
For each $i \in I$:
- $x_i$: Number of units of product $i$ to fulfill  
  Domain: $x_i \in \mathbb{Z}_+$ (non-negative integers)

**Objective:**  
Maximize total revenue:
$$
\max \sum_{i \in I} A_i x_i
$$

**Constraints:**
1. **Demand fulfillment:**  
  $\forall i \in I: \quad x_i \leq d_i$

2. **Inventory limit:**  
  $\forall i \in I: \quad x_i \leq I_i$

3. **Non-negativity and integrality:**  
  $\forall i \in I: \quad x_i \in \mathbb{Z}_+, \ x_i \geq 0$

---

#### Data Mapping

- **Source Table:**  
  `/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM15/SampleSalesData.csv`

- **Columns Used:**  
  - Product Name (for index set $I$, filtered by prefix ‘S700_’)  
  - Revenue (parameter $A_i$)  
  - Demand (parameter $d_i$)  
  - Initial Inventory (parameter $I_i$)