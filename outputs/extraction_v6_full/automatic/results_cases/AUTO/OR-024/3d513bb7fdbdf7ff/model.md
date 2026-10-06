#### Abstract Mathematical Model

**Index Set:**
- $I$ : Set of all products with identifiers starting with ‘S700_’, indexed by $i$.

**Parameters:**
- $p_i$ : Revenue per unit of product $i$.  
- $d_i$ : Total demand for product $i$ over the sales horizon.  
- $s_i$ : Initial inventory of product $i$.

**Decision Variables:**
- $x_i$ : Number of units of product $i$ to fulfill, $\forall i \in I$.

**Objective:**
\[
\max \sum_{i \in I} p_i \cdot x_i
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

- $I$ : All rows in table_id file_0_view_0 where [Product Name] has prefix ‘S700_’.
- $p_i$ : [Revenue] column in table_id file_0_view_0, indexed by [Product Name].
- $d_i$ : [Demand] column in table_id file_0_view_0, indexed by [Product Name].
- $s_i$ : [Initial Inventory] column in table_id file_0_view_0, indexed by [Product Name].

(CSVQA_DATA bindings: see table_id file_0_view_0, columns [Product Name], [Revenue], [Demand], [Initial Inventory], filtered by prefix ‘S700_’ on [Product Name].)