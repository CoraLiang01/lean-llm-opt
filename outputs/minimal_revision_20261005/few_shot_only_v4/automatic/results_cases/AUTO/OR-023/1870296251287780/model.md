#### Sets

- $I$: Set of all products classified under ‘ELE-S’, indexed by $i$.

#### Parameters

- $r_i$: Revenue per unit of product $i$ (from column "Revenue").
- $d_i$: Total deterministic demand for product $i$ (from column "Demand").
- $s_i$: Initial inventory of product $i$ (from column "Initial Inventory").

#### Decision Variables

- $x_i$: Number of units of product $i$ to fulfill, $\forall i \in I$.

#### Objective

$$
\max \sum_{i \in I} r_i x_i
$$

#### Constraints

1. **Inventory constraint:**  
   $$
   x_i \leq s_i, \quad \forall i \in I
   $$

2. **Demand constraint:**  
   $$
   x_i \leq d_i, \quad \forall i \in I
   $$

3. **Non-negativity and integrality:**  
   $$
   x_i \in \mathbb{Z}_+, \quad \forall i \in I
   $$

---

#### Data Mapping

- **Table:** `/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM14/SalesStoreoverview.csv`
- **Columns:**  
  - Product index: `"Product_Reference"`  
  - Revenue: `"Revenue"`  
  - Demand: `"Demand"`  
  - Initial inventory: `"Initial Inventory"`