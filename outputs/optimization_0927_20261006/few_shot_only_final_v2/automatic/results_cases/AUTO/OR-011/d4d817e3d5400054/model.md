**Sets:**
- $I$ : Set of all products classified under ‘id999’.

**Parameters:**
- $A_i$ : Revenue per unit of product $i \in I$ (from column ‘Revenue’).
- $d_i$ : Demand for product $i \in I$ during the sales horizon (from column ‘Demand’).
- $I_i$ : Initial inventory of product $i \in I$ (from column ‘Initial Inventory’).

**Decision Variables:**
- $x_i$ : Number of units of product $i \in I$ to fulfill, $x_i \in \mathbb{Z}_+$.

**Objective:**
\[
\max \sum_{i \in I} A_i x_i
\]

**Constraints:**
1. **Inventory constraint:**  
   \[
   x_i \leq I_i \quad \forall i \in I
   \]
2. **Demand constraint:**  
   \[
   x_i \leq d_i \quad \forall i \in I
   \]
3. **Non-negativity and integrality:**  
   \[
   x_i \in \mathbb{Z}_+, \quad \forall i \in I
   \]

---

**Data Mapping:**

- Table: `file_0_view_0` (from `OnlineRetailSalesDataset.csv`)
    - Product identifier: `id_number` (select those with value ‘id999’)
    - Revenue per unit: `Revenue`
    - Demand: `Demand`
    - Initial Inventory: `Initial Inventory`