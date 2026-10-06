#### Abstract Mathematical Model

**Index Set:**
- $I$ : Set of all ‘TABLET’ smartphone models, indexed by $i$.

**Parameters:**
- $A_i$ : Revenue per unit for model $i$.
- $I_i$ : Initial inventory for model $i$.
- $d_i$ : Demand for model $i$.

**Decision Variables:**
- $x_i$ : Number of units of model $i$ to fulfill, $\forall i \in I$.

**Objective:**
\[
\max \sum_{i \in I} A_i \cdot x_i
\]

**Constraints:**
1. Inventory constraint:
   \[
   x_i \leq I_i, \quad \forall i \in I
   \]
2. Demand constraint:
   \[
   x_i \leq d_i, \quad \forall i \in I
   \]
3. Non-negativity and integrality:
   \[
   x_i \in \mathbb{Z}_+, \quad \forall i \in I
   \]

---

#### Data Mapping

- $I$ : All rows in table_id = file_0_view_0 where [Product Name] has prefix 'TABLET'.
- $A_i$ : [Revenue] column in table_id = file_0_view_0, indexed by [Product Name].
- $I_i$ : [Initial Inventory] column in table_id = file_0_view_0, indexed by [Product Name].
- $d_i$ : [Demand] column in table_id = file_0_view_0, indexed by [Product Name].

(CSVQA_DATA bindings: see table_id = file_0_view_0, columns [Product Name], [Revenue], [Initial Inventory], [Demand], filtered by [Product Name] prefix 'TABLET'.)