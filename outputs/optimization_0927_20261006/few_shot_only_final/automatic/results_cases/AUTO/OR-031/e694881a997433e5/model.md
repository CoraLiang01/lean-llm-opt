**Sets:**  
Let $\mathcal{I}$ be the set of all dairy products, indexed by $i$.

**Parameters:**  
- $A_i$: Revenue per unit of product $i$ (from column "Revenue")  
- $d_i$: Demand for product $i$ (from column "Demand")  
- $I_i$: Initial inventory of product $i$ (from column "Initial Inventory")  

**Decision Variables:**  
- $x_i$: Number of units of product $i$ to fulfill, $\forall i \in \mathcal{I}$

**Objective:**  
\[
\max \sum_{i \in \mathcal{I}} A_i \cdot x_i
\]

**Constraints:**  
1. **Inventory Constraint:**  
   \[
   x_i \leq I_i, \quad \forall i \in \mathcal{I}
   \]
2. **Demand Constraint:**  
   \[
   x_i \leq d_i, \quad \forall i \in \mathcal{I}
   \]
3. **Non-negativity and Integrality:**  
   \[
   x_i \in \mathbb{Z}_+, \quad \forall i \in \mathcal{I}
   \]

---

**Data Mapping:**  
- Table: `file_0_view_0` (from `DairyGoodsSalesDataset.csv`)
    - Product index set $\mathcal{I}$: column `"Full_Product_Name"`
    - Revenue parameter $A_i$: column `"Revenue"`
    - Demand parameter $d_i$: column `"Demand"`
    - Initial inventory parameter $I_i$: column `"Initial Inventory"`