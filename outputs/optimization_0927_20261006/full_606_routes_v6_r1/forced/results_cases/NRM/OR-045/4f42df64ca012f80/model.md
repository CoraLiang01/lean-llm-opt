#### Abstract Mathematical Model

**Index Sets:**
- $I$: Set of produce types (indexed by $i$).

**Parameters:**
- $v_i$: Benefit per unit of produce type $i$. (from products.csv, column "Value")
- $w_i$: Weight per unit of produce type $i$. (from products.csv, column "Weight")
- $C$: Overall inventory capacity. (from capacity.csv, column "Capacity")

**Decision Variables:**
- $x_i \in \mathbb{Z}_{\geq 0}$: Number of units of produce type $i$ to order daily.

**Objective:**
\[
\max \sum_{i \in I} v_i x_i
\]

**Constraints:**
1. **Capacity Constraint:**
   \[
   \sum_{i \in I} w_i x_i \leq C
   \]
2. **Integrality:**
   \[
   x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i \in I
   \]

---

#### Data Mapping

- **products.csv**
  - Table ID: file_1_view_0
    - "ProductName": Index set $I$
    - "Value": Parameter $v_i$
    - "Weight": Parameter $w_i$
- **capacity.csv**
  - Table ID: file_0_view_0
    - "Capacity": Parameter $C$