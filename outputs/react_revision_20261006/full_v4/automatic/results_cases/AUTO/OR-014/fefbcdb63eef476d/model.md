#### Mathematical Optimization Model

**Index Set:**
- $I$: set of all pizza types (from all "Product Name" entries in table_id file_0_view_0).

**Parameters:**
- $A_i$: revenue per unit of pizza type $i$ (from "Revenue", table_id file_0_view_0).
- $d_i$: total demand for pizza type $i$ (from "Demand", table_id file_0_view_0).
- $I_i$: initial inventory for pizza type $i$ (from "Initial Inventory", table_id file_0_view_0).

**Decision Variables:**
- $x_i$: number of units of pizza type $i$ to fulfill, $x_i \in \mathbb{Z}_+, \forall i \in I$.

**Objective:**
\[
\max \sum_{i \in I} A_i x_i
\]

**Constraints:**
\[
x_i \leq d_i, \quad \forall i \in I
\]
\[
x_i \leq I_i, \quad \forall i \in I
\]
\[
x_i \in \mathbb{Z}_+, \quad \forall i \in I
\]

#### Data Mapping

- $I$: All "Product Name" in table_id file_0_view_0, column "Product Name"
- $A_i$: table_id file_0_view_0, column "Revenue"
- $d_i$: table_id file_0_view_0, column "Demand"
- $I_i$: table_id file_0_view_0, column "Initial Inventory"