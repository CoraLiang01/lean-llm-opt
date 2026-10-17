#### Abstract Mathematical Optimization Model

**Index Set:**
- $I$: Set of all products classified as ‘27in’ (from Product Name column, prefix "27in").

**Parameters:**
- $A_i$: Revenue per unit of product $i \in I$ (from Revenue column).
- $d_i$: Demand for product $i \in I$ (from Demand column).
- $I_i$: Initial inventory for product $i \in I$ (from Initial Inventory column).

**Decision Variables:**
- $x_i$: Number of units of product $i \in I$ to fulfill, $x_i \in \mathbb{Z}_+$ (non-negative integers).

**Objective:**
\[
\max \sum_{i \in I} A_i \cdot x_i
\]

**Constraints:**
1. **Inventory Constraint:** 
   \[
   x_i \leq I_i \quad \forall i \in I
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

- **Source Table:** SalesDataAnalysis.csv
- **Index Set $I$:** All records where [Product Name] has prefix "27in" (CSVQA filter: Product Name prefix "27in")
- **Parameter $A_i$:** [Revenue] column, table_id: file_0_view_0
- **Parameter $d_i$:** [Demand] column, table_id: file_0_view_0
- **Parameter $I_i$:** [Initial Inventory] column, table_id: file_0_view_0

No literal record values or counts are included; all data is referenced symbolically as required.