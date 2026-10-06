#### Abstract Mathematical Model

**Index Set:**
- $i \in \mathcal{B}$: Set of all products classified under ‘Books’ (indexed by Product_Name).

**Parameters:**
- $A_i$: Revenue per unit of product $i$ (from column Revenue, table_id file_0_view_0).
- $d_i$: Demand for product $i$ (from column Demand, table_id file_0_view_0).
- $I_i$: Initial inventory for product $i$ (from column Initial Inventory, table_id file_0_view_0).

**Decision Variables:**
- $x_i$: Number of units of product $i$ to fulfill, $x_i \in \mathbb{Z}_+$, $\forall i \in \mathcal{B}$.

**Objective:**
\[
\max \sum_{i \in \mathcal{B}} A_i \cdot x_i
\]

**Constraints:**
1. **Inventory Constraint:** 
   \[
   x_i \leq I_i, \quad \forall i \in \mathcal{B}
   \]
2. **Demand Constraint:** 
   \[
   x_i \leq d_i, \quad \forall i \in \mathcal{B}
   \]
3. **Non-negativity and Integrality:**
   \[
   x_i \in \mathbb{Z}_+, \quad \forall i \in \mathcal{B}
   \]

---

#### Data Mapping

- $\mathcal{B}$ (index set $i$): All rows in table_id file_0_view_0 where Product_Name has prefix "Books_".
- $A_i$: file_0_view_0, column "Revenue", indexed by Product_Name.
- $d_i$: file_0_view_0, column "Demand", indexed by Product_Name.
- $I_i$: file_0_view_0, column "Initial Inventory", indexed by Product_Name.