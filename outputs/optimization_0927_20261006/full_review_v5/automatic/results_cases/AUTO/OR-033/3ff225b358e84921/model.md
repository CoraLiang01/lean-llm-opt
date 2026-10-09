#### Abstract Mathematical Model

**Index Sets:**
- $I$: Set of all products classified under ‘Baby’ (indexed by $i$).

**Parameters:**
- $A_i$: Revenue per unit for product $i \in I$ (from column ‘Revenue’).
- $I_i$: Initial inventory for product $i \in I$ (from column ‘Initial Inventory’).
- $d_i$: Demand for product $i \in I$ (from column ‘Demand’).

**Decision Variables:**
- $x_i$: Number of units of product $i \in I$ to fulfill, $x_i \in \mathbb{Z}_+$.

**Objective:**
\[
\max \sum_{i \in I} A_i \cdot x_i
\]

**Constraints:**
1. Inventory constraint:
   \[
   x_i \leq I_i \quad \forall i \in I
   \]
2. Demand constraint:
   \[
   x_i \leq d_i \quad \forall i \in I
   \]
3. Non-negativity and integrality:
   \[
   x_i \in \mathbb{Z}_+, \quad \forall i \in I
   \]

---

#### Data Mapping

- Source Table: EuropeSalesRecords.csv (table_id: file_0_view_0)
- Filter: Rows where ‘Product Name’ has prefix ‘Baby’
- Columns used:
    - ‘Product Name’ (for index set $I$)
    - ‘Revenue’ (parameter $A_i$)
    - ‘Initial Inventory’ (parameter $I_i$)
    - ‘Demand’ (parameter $d_i$)