---

### Abstract Mathematical Model

#### Index Sets

- $I$: Set of all products, indexed by $i$.

#### Parameters

- $A_i$: Revenue per unit of product $i$.  
- $d_i$: Total demand for product $i$.  
- $I_i$: Initial inventory available for product $i$.

#### Decision Variables

- $x_i$: Number of units of product $i$ to fulfill (integer, $x_i \geq 0$).

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
   x_i \in \mathbb{Z}_{+} \quad \forall i \in I
   \]

---

### Data Mapping

- **Table:** `file_0_view_0` (from `RetailSalesDataset.csv`)
- **Columns:**
    - Product Name: defines set $I$
    - Revenue: parameter $A_i$
    - Demand: parameter $d_i$
    - Initial Inventory: parameter $I_i$

---