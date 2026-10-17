#### Abstract Mathematical Optimization Model

**Index Sets:**
- $I$: Set of all baked goods (indexed by $i$).

**Parameters:**
- $a_i$: Revenue per unit of baked good $i$ (from column ‘Revenue’).
- $d_i$: Total demand for baked good $i$ (from column ‘Demand’).
- $s_i$: Initial inventory for baked good $i$ (from column ‘Initial Inventory’).

**Decision Variables:**
- $x_i$: Quantity of baked good $i$ to fulfill, $\forall i \in I$.

**Objective:**
\[
\max \sum_{i \in I} a_i \cdot x_i
\]

**Constraints:**
1. **Demand fulfillment:** 
   \[
   x_i \leq d_i, \quad \forall i \in I
   \]
2. **Inventory availability:** 
   \[
   x_i \leq s_i, \quad \forall i \in I
   \]
3. **Non-negativity and integrality:** 
   \[
   x_i \in \mathbb{Z}_+, \quad \forall i \in I
   \]

---

#### Data Mapping

- **Source Table:** `file_0_view_0` (from `Frenchbakerydailysales.csv`)
- **Index Set $I$:** All rows in column `Product Name`
- **Parameter $a_i$:** Column `Revenue`
- **Parameter $d_i$:** Column `Demand`
- **Parameter $s_i$:** Column `Initial Inventory`