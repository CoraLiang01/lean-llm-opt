#### Abstract Mathematical Model

**Index Set:**
- $I$: set of all clothing products (indexed by $i$).

**Parameters:**
- $A_i$: revenue per unit of product $i$ (from column ‘Revenue’).
- $d_i$: total demand for product $i$ (from column ‘Demand’).
- $I_i$: initial inventory for product $i$ (from column ‘Initial Inventory’).

**Decision Variables:**
- $x_i$: number of units of product $i$ to fulfill, $x_i \in \mathbb{Z}_+, \forall i \in I$.

**Objective:**
\[
\max \sum_{i \in I} A_i x_i
\]

**Constraints:**
1. **Inventory constraint:** 
   \[
   x_i \leq I_i, \quad \forall i \in I
   \]
2. **Demand constraint:** 
   \[
   x_i \leq d_i, \quad \forall i \in I
   \]
3. **Non-negativity and integrality:** 
   \[
   x_i \in \mathbb{Z}_+, \quad \forall i \in I
   \]

---

#### Data Mapping

- **Source Table:** `file_0_view_0` (from `Salesofsummerclothes.csv`)
- **Columns Used:**
  - Product Name: index set $I$
  - Revenue: parameter $A_i$
  - Demand: parameter $d_i$
  - Initial Inventory: parameter $I_i$
- **Selection:** All records in source order; no filters applied.