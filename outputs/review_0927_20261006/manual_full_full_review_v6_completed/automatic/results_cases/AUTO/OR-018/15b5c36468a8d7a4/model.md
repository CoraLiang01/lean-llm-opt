#### Abstract Mathematical Model

**Index Sets:**
- $I$: Set of all products classified under ‘Baby’.

**Parameters:**
- $A_i$: Revenue per unit for product $i \in I$ (from column ‘Revenue’).
- $d_i$: Total demand for product $i \in I$ (from column ‘Demand’).
- $I_i$: Initial inventory for product $i \in I$ (from column ‘Initial Inventory’).

**Decision Variables:**
- $x_i$: Number of units of product $i \in I$ to fulfill, $x_i \in \mathbb{Z}_+$.

**Objective:**
\[
\max \sum_{i \in I} A_i \cdot x_i
\]

**Constraints:**
1. **Inventory Constraint:** 
   \[
   x_i \leq I_i, \quad \forall i \in I
   \]
2. **Demand Constraint:** 
   \[
   x_i \leq d_i, \quad \forall i \in I
   \]
3. **Non-negativity and Integrality:**
   \[
   x_i \in \mathbb{Z}_+, \quad \forall i \in I
   \]

---

**Data Mapping:**

- **Source Table:** `Salesdata.csv`
- **Table ID:** `file_0_view_0`
- **Columns Used:**
  - `Product Name` (filtered: prefix = "Baby")
  - `Revenue`
  - `Demand`
  - `Initial Inventory`
- **Filter Applied:** Only rows where `Product Name` starts with "Baby" (as classified under ‘Baby’ in the query).
- **Returned Rows:** All such rows, preserving source order and identifiers.