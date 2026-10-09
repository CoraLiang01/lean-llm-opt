#### Abstract Mathematical Model

**Index Set:**

- $I$ : Set of all products in the source table whose name contains the substring “FAUX”.

**Parameters:**

- $a_i$ : Revenue per unit of product $i \in I$ (from column “Revenue”).
- $d_i$ : Total demand for product $i \in I$ (from column “Demand”).
- $s_i$ : Initial inventory for product $i \in I$ (from column “Initial Inventory”).

**Decision Variables:**

- $x_i$ : Number of units of product $i \in I$ to fulfill, $x_i \in \mathbb{Z}_+$ (non-negative integers).

**Objective:**

\[
\max \quad \sum_{i \in I} a_i \, x_i
\]

**Constraints:**

1. **Demand fulfillment:**
   \[
   x_i \leq d_i \qquad \forall i \in I
   \]

2. **Inventory availability:**
   \[
   x_i \leq s_i \qquad \forall i \in I
   \]

3. **Non-negativity and integrality:**
   \[
   x_i \in \mathbb{Z}_+, \qquad \forall i \in I
   \]

---

#### Data Mapping

- **Table:** `file_0_view_0` (from `ZARASales.csv`)
- **Index Set $I$:** All rows where `Product Name` contains the substring “FAUX”
- **Parameter $a_i$:** Column `Revenue`
- **Parameter $d_i$:** Column `Demand`
- **Parameter $s_i$:** Column `Initial Inventory`