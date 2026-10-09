#### Abstract Mathematical Optimization Model

**Index Set:**
- $I$: Set of products (indexed by $i$).

**Parameters:**
- $A_i$: Revenue per unit of product $i$.  
- $d_i$: Total demand for product $i$ over the sales horizon.  
- $I_i$: Initial inventory available for product $i$.

**Decision Variables:**
- $x_i$: Number of units of product $i$ to fulfill (integer, $x_i \geq 0$).

**Objective:**
\[
\max \sum_{i \in I} A_i \cdot x_i
\]

**Constraints:**
1. **Demand fulfillment:**  
   \[
   x_i \leq d_i \quad \forall i \in I
   \]
2. **Inventory limit:**  
   \[
   x_i \leq I_i \quad \forall i \in I
   \]
3. **Non-negativity and integrality:**  
   \[
   x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
   \]

---

#### Data Mapping

- **Table:** file_0_view_0 (from nike_Shoes_Sales.csv)
- **Columns:**
    - Product Name $\rightarrow$ index set $I$
    - Revenue $\rightarrow$ parameter $A_i$
    - Demand $\rightarrow$ parameter $d_i$
    - Initial Inventory $\rightarrow$ parameter $I_i$
- **Selection:** All records returned by CSVQA (no additional filtering applied).