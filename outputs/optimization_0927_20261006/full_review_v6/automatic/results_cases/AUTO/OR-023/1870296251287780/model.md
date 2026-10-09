#### Abstract Mathematical Optimization Model

**Index Set:**
- $I$: Set of all products classified under ‘ELE-S’ (indexed by $i$).

**Parameters:**
- $r_i$: Revenue per unit of product $i$ (from column ‘Revenue’).
- $d_i$: Demand for product $i$ (from column ‘Demand’).
- $s_i$: Initial inventory of product $i$ (from column ‘Initial Inventory’).

**Decision Variables:**
- $x_i$: Number of units of product $i$ to fulfill, $\forall i \in I$.

**Objective:**
\[
\max \sum_{i \in I} r_i \cdot x_i
\]

**Constraints:**
1. **Inventory Constraint:** 
   \[
   x_i \leq s_i, \quad \forall i \in I
   \]
2. **Demand Constraint:** 
   \[
   x_i \leq d_i, \quad \forall i \in I
   \]
3. **Non-negativity and Integrality:**
   \[
   x_i \in \mathbb{Z}_+, \quad \forall i \in I
   \]

---

#### Data Mapping

- **Table:** file_0_view_0 (from SalesStoreoverview.csv)
- **Index Set:** $I$ = all records where Product_Reference has prefix ‘ELE-S’
- **Parameters:**
    - $r_i$: column ‘Revenue’
    - $d_i$: column ‘Demand’
    - $s_i$: column ‘Initial Inventory’
- **Decision Variables:** $x_i$ for each $i \in I$