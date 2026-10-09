#### Mathematical Optimization Model

**Index Set:**  
Let $I$ be the set of all products, indexed by $i$.

**Parameters:**  
For each $i \in I$:
- $A_i$: Revenue per unit of product $i$ (from column "Revenue")
- $d_i$: Demand for product $i$ during the sales cycle (from column "Demand")
- $I_i$: Initial inventory of product $i$ (from column "Initial Inventory")

**Decision Variables:**  
For each $i \in I$:
- $x_i$: Number of orders fulfilled for product $i$  
  $x_i \in \mathbb{Z}_+, \quad \forall i \in I$

**Objective:**  
$\max \sum_{i \in I} A_i x_i$

**Constraints:**
- Inventory constraint: $x_i \leq I_i, \quad \forall i \in I$
- Demand constraint: $x_i \leq d_i, \quad \forall i \in I$
- Non-negativity and integrality: $x_i \in \mathbb{Z}_+, \quad \forall i \in I$

---

**Data Mapping:**  
- Table: `file_0_view_0` (from `MobileSalesDataset.csv`)
    - Product index set $I$: All rows, column "Product Name"
    - Revenue $A_i$: column "Revenue"
    - Demand $d_i$: column "Demand"
    - Initial inventory $I_i$: column "Initial Inventory"