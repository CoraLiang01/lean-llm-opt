#### Abstract Optimization Model

**Index Sets:**
- $I$: Set of all products classified under ‘Baby’ (from column ‘Product Name’ with prefix "Baby").

**Parameters:**
- $A_i$: Revenue per unit of product $i \in I$ (from column ‘Revenue’ in table_id: file_0_view_0).
- $d_i$: Total demand for product $i \in I$ (from column ‘Demand’ in table_id: file_0_view_0).
- $I_i$: Initial inventory for product $i \in I$ (from column ‘Initial Inventory’ in table_id: file_0_view_0).

**Decision Variables:**
- $x_i$: Number of units of product $i \in I$ to fulfill; $x_i \in \mathbb{Z}_+, \forall i \in I$.

**Objective:**
\[
\max \sum_{i \in I} A_i \cdot x_i
\]

**Constraints:**
1. Inventory and Demand Fulfillment:
   \[
   0 \leq x_i \leq \min\{d_i, I_i\}, \quad \forall i \in I
   \]
   (Equivalently, two constraints per $i$: $x_i \leq d_i$ and $x_i \leq I_i$.)

2. Integer Variables:
   \[
   x_i \in \mathbb{Z}_+, \quad \forall i \in I
   \]

---

**Data Mapping:**

- Source Table: file_0_view_0 (from Salesdata.csv)
- Filter: All records where ‘Product Name’ has prefix "Baby" (i.e., classified under ‘Baby’)
- Columns used:
    - ‘Product Name’ (index set $I$)
    - ‘Revenue’ (parameter $A_i$)
    - ‘Demand’ (parameter $d_i$)
    - ‘Initial Inventory’ (parameter $I_i$)