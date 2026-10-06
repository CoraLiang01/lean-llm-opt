#### Abstract Mathematical Model

**Index Sets:**
- $I$: Set of all products classified as 'Fashion' in the dataset.

**Parameters:**
- $a_i$: Revenue per unit of product $i \in I$ (from column 'Revenue').
- $d_i$: Deterministic demand for product $i \in I$ (from column 'Demand').
- $s_i$: Initial inventory for product $i \in I$ (from column 'Initial Inventory').

**Decision Variables:**
- $x_i$: Number of units of product $i \in I$ to fulfill, $x_i \in \mathbb{Z}_+$.

**Objective:**
\[
\max \sum_{i \in I} a_i x_i
\]

**Constraints:**
1. **Inventory Constraint:**  
   $\forall i \in I: \quad x_i \leq s_i$
2. **Demand Constraint:**  
   $\forall i \in I: \quad x_i \leq d_i$
3. **Nonnegativity and Integrality:**  
   $\forall i \in I: \quad x_i \geq 0,\quad x_i \in \mathbb{Z}$

---

#### Data Mapping

- **Source Table:**  
  `/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM17/SupermarketSales.csv`
- **Columns Used:**  
  - Product Name (for identifying 'Fashion' products and index set $I$)
  - Revenue (parameter $a_i$)
  - Demand (parameter $d_i$)
  - Initial Inventory (parameter $s_i$)