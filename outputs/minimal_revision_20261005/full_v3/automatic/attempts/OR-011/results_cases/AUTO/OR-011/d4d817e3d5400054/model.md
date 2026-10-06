#### Abstract Mathematical Model

**Index Sets**

- $I$ : Set of products classified under ‘id999’.

**Parameters**

- $A_i$ : Revenue per unit of product $i \in I$ (from column ‘Revenue’ in table_id: file_0_view_0).
- $d_i$ : Demand for product $i \in I$ during the sales horizon (from column ‘Demand’ in table_id: file_0_view_0).
- $I_i$ : Initial inventory of product $i \in I$ (from column ‘Initial Inventory’ in table_id: file_0_view_0).

**Decision Variables**

- $x_i$ : Number of units of product $i \in I$ to fulfill, $x_i \in \mathbb{Z}_+, \forall i \in I$.

**Objective**

\[
\max \quad \sum_{i \in I} A_i \cdot x_i
\]

**Constraints**

1. **Inventory Constraints:**  
   \[
   x_i \leq I_i, \quad \forall i \in I
   \]

2. **Demand Constraints:**  
   \[
   x_i \leq d_i, \quad \forall i \in I
   \]

3. **Non-negativity and Integrality:**  
   \[
   x_i \in \mathbb{Z}_+, \quad \forall i \in I
   \]

---

#### Data Mapping

- **table_id:** file_0_view_0
    - **id_number:** Used to select products classified under ‘id999’.
    - **Revenue:** Parameter $A_i$ (revenue per unit).
    - **Demand:** Parameter $d_i$ (demand per product).
    - **Initial Inventory:** Parameter $I_i$ (initial inventory per product).