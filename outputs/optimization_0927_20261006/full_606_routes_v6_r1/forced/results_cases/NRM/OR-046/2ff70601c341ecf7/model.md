#### Abstract Mathematical Model

**Index Sets:**
- $P$: set of products (from products.csv), indexed by $i$

**Parameters:**
- $v_i$: value/benefit per unit of product $i$ (from products.csv, column "Value")
- $w_i$: weight per unit of product $i$ (from products.csv, column "Weight")
- $C$: overall stock capacity (from capacity.csv, column "Capacity")

**Decision Variables:**
- $x_i$: number of units of product $i$ to order each day, $x_i \geq 0$, integer, $\forall i \in P$

**Objective:**
\[
\max \sum_{i \in P} v_i \cdot x_i
\]

**Constraints:**
1. **Stock Capacity Constraint:**
   \[
   \sum_{i \in P} w_i \cdot x_i \leq C
   \]
2. **Non-negativity and Integrality:**
   \[
   x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i \in P
   \]

---

#### Data Mapping

- **products.csv** (`file_1_view_0`):
  - Product index set $P$ from column "ProductName"
  - Parameter $v_i$ from column "Value"
  - Parameter $w_i$ from column "Weight"
- **capacity.csv** (`file_0_view_0`):
  - Parameter $C$ from column "Capacity" (single record, overall constraint)