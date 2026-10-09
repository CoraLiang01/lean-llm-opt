---

### Abstract Mathematical Model

#### Index Sets
- Let $\mathcal{I}$ be the set of all products $i$ such that the value in column `Product Name` contains the substring `'FAUX'` in table `file_0_view_0`.

#### Parameters
- $A_i$: Revenue per unit of product $i$, from column `Revenue` in table `file_0_view_0`.
- $d_i$: Total demand for product $i$, from column `Demand` in table `file_0_view_0`.
- $I_i$: Initial inventory for product $i$, from column `Initial Inventory` in table `file_0_view_0`.

#### Decision Variables
- $x_i$: Number of units of product $i$ to fulfill, for all $i \in \mathcal{I}$.

#### Objective
\[
\max \sum_{i \in \mathcal{I}} A_i \cdot x_i
\]

#### Constraints
1. **Inventory Constraint:**  
   \[
   x_i \leq I_i \qquad \forall i \in \mathcal{I}
   \]
2. **Demand Constraint:**  
   \[
   x_i \leq d_i \qquad \forall i \in \mathcal{I}
   \]
3. **Nonnegativity and Integrality:**  
   \[
   x_i \in \mathbb{Z}_+, \qquad \forall i \in \mathcal{I}
   \]

---

### Data Mapping

- **Source Table:** `file_0_view_0` (from `ZARASales.csv`)
- **Selection Rule:** $\mathcal{I} = \{i : \text{`Product Name`} \text{ contains the substring } 'FAUX'\}$
- **Parameters:**
    - $A_i$ from column `Revenue`
    - $d_i$ from column `Demand`
    - $I_i$ from column `Initial Inventory`
- **Variables:** $x_i$ for each $i \in \mathcal{I}$

---