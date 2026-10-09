#### Abstract Mathematical Model

**Index Sets:**
- $I$: Set of all products classified as ‘Fashion’ (indexed by $i$).

**Parameters:**
- $r_i$: Revenue per unit of product $i$ (from column [Revenue]).
- $d_i$: Total demand for product $i$ over the sales horizon (from column [Demand]).
- $s_i$: Initial inventory of product $i$ (from column [Initial Inventory]).

**Decision Variables:**
- $x_i$: Number of units of product $i$ to fulfill, $x_i \in \mathbb{Z}_+$ (non-negative integers), for all $i \in I$.

**Objective:**
\[
\max \sum_{i \in I} r_i \cdot x_i
\]

**Constraints:**
1. **Inventory Constraint:** 
   \[
   x_i \leq s_i \quad \forall i \in I
   \]
2. **Demand Constraint:** 
   \[
   x_i \leq d_i \quad \forall i \in I
   \]
3. **Non-negativity and Integrality:**
   \[
   x_i \in \mathbb{Z}_+, \quad \forall i \in I
   \]

---

#### Data Mapping

- **Table:** `SupermarketSales.csv`
- **Table ID:** `file_0_view_0`
- **Filter Applied:** Rows where [Product Name] has prefix "Fashion"
- **Columns Used:**
  - [Product Name] $\rightarrow$ Index set $I$
  - [Revenue] $\rightarrow$ Parameter $r_i$
  - [Initial Inventory] $\rightarrow$ Parameter $s_i$
  - [Demand] $\rightarrow$ Parameter $d_i$