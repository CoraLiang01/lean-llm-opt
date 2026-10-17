#### Abstract Mathematical Model

**Index Sets:**
- $I$: Set of drug products (from products.csv, column ProductName)

**Parameters:**
- $v_i$: Benefit per unit of drug $i \in I$ (from products.csv, column Value)
- $w_i$: Stock weight per unit of drug $i \in I$ (from products.csv, column Weight)
- $C$: Overall stock capacity (from capacity.csv, column Capacity)

**Decision Variables:**
- $x_i$: Number of units of drug $i \in I$ to order each day, $x_i \in \mathbb{Z}_{\geq 0}$

**Objective:**
\[
\max \sum_{i \in I} v_i x_i
\]

**Constraints:**
1. **Stock Capacity Constraint:**
   \[
   \sum_{i \in I} w_i x_i \leq C
   \]
2. **Non-negativity and Integrality:**
   \[
   x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i \in I
   \]

---

#### Data Mapping

- **products.csv**
  - ProductName: Index set $I$
  - Value: Parameter $v_i$
  - Weight: Parameter $w_i$
- **capacity.csv**
  - Capacity: Parameter $C$