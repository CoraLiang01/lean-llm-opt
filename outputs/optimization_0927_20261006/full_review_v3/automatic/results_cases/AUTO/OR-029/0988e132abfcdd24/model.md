### Abstract Mathematical Model

#### Index Sets
- $I$: Set of all products classified as ‘FAUX’ (indexed by $i$).

#### Parameters
- $A_i$: Revenue per unit of product $i$ (from column ‘Revenue’).
- $d_i$: Total demand for product $i$ (from column ‘Demand’).
- $I_i$: Initial inventory for product $i$ (from column ‘Initial Inventory’).

#### Decision Variables
- $x_i$: Number of units of product $i$ to fulfill, $x_i \in \mathbb{Z}_+$ (non-negative integers).

#### Objective
\[
\max \sum_{i \in I} A_i \cdot x_i
\]

#### Constraints
1. **Inventory Constraint:** 
   \[
   x_i \leq I_i \quad \forall i \in I
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

### Data Mapping

- **Table:** file_0_view_0 (from ZARASales.csv)
- **Filter:** Product Name starts with ‘FAUX’
- **Columns:**
    - Product Name $\rightarrow$ Index set $I$
    - Revenue $\rightarrow$ Parameter $A_i$
    - Demand $\rightarrow$ Parameter $d_i$
    - Initial Inventory $\rightarrow$ Parameter $I_i$