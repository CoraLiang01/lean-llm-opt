#### Abstract Mathematical Model

**Index Sets:**

- $I$: Set of all products classified under 'FAUX'.

**Parameters:**

- $a_i$: Revenue per unit of product $i \in I$ (from column 'Revenue').
- $d_i$: Demand for product $i \in I$ (from column 'Demand').
- $s_i$: Initial inventory of product $i \in I$ (from column 'Initial Inventory').

**Decision Variables:**

- $x_i$: Number of units of product $i \in I$ to fulfill, $x_i \in \mathbb{Z}_+$.

**Objective:**

$$
\max \sum_{i \in I} a_i x_i
$$

**Constraints:**

1. **Demand fulfillment:**  
   $x_i \leq d_i \quad \forall i \in I$

2. **Inventory availability:**  
   $x_i \leq s_i \quad \forall i \in I$

3. **Non-negativity and integrality:**  
   $x_i \in \mathbb{Z}_+, \quad \forall i \in I$

---

#### Data Mapping

- **Source Table:** `/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM20/ZARASales.csv`
- **Columns Used:**
  - `Product Name` (for index set $I$, filtered to 'FAUX' products)
  - `Revenue` (parameter $a_i$)
  - `Demand` (parameter $d_i$)
  - `Initial Inventory` (parameter $s_i$)