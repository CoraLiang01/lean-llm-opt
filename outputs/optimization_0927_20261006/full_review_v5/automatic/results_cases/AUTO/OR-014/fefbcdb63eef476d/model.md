#### Abstract Mathematical Optimization Model

**Index Sets:**

- $I$ : Set of all pizza types (indexed by $i$).

**Parameters:**

- $A_i$ : Revenue per unit for pizza type $i$.  
- $d_i$ : Total deterministic demand for pizza type $i$ over the sales horizon.  
- $I_i$ : Initial inventory available for pizza type $i$.

**Decision Variables:**

- $x_i$ : Number of units of pizza type $i$ to fulfill (integer, $x_i \geq 0$).

---

**Objective:**

$$
\max \sum_{i \in I} A_i \cdot x_i
$$

**Constraints:**

1. **Inventory Constraints:**  
   $$
   x_i \leq I_i \quad \forall i \in I
   $$

2. **Demand Constraints:**  
   $$
   x_i \leq d_i \quad \forall i \in I
   $$

3. **Non-negativity and Integrality:**  
   $$
   x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
   $$

---

#### Data Mapping

- **Table:** PizzaSalesDataset.csv
- **table_id:** file_0_view_0
- **Columns:**
    - Product Name $\rightarrow$ index set $I$
    - Revenue $\rightarrow$ parameter $A_i$
    - Demand $\rightarrow$ parameter $d_i$
    - Initial Inventory $\rightarrow$ parameter $I_i$

All pizza types and their associated parameters are included as returned by the query. No additional filters or restrictions are applied.