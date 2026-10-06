#### Abstract Mathematical Model

**Index Set:**  
Let $I$ be the set of all products classified under ‘FAUX’.

**Parameters:**  
For each $i \in I$:
- $r_i$: Revenue per unit of product $i$ (from column ‘Revenue’)
- $d_i$: Demand for product $i$ (from column ‘Demand’)
- $s_i$: Initial inventory of product $i$ (from column ‘Initial Inventory’)

**Decision Variables:**  
For each $i \in I$:
- $x_i$: Number of units of product $i$ to fulfill  
  Domain: $x_i \in \mathbb{Z}_+$ (non-negative integers)

**Objective:**  
Maximize total revenue:
$$
\max \sum_{i \in I} r_i x_i
$$

**Constraints:**
1. **Inventory constraint:**  
  For all $i \in I$,
$$
x_i \leq s_i
$$

2. **Demand constraint:**  
  For all $i \in I$,
$$
x_i \leq d_i
$$

3. **Non-negativity and integrality:**  
  For all $i \in I$,
$$
x_i \in \mathbb{Z}_+, \quad x_i \geq 0
$$

---

#### Data Mapping

- **Source Table:** `/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM20/ZARASales.csv`
- **Columns Used:**
  - `Product Name` (for index set $I$ and product identification)
  - `Revenue` (parameter $r_i$)
  - `Demand` (parameter $d_i$)
  - `Initial Inventory` (parameter $s_i$)