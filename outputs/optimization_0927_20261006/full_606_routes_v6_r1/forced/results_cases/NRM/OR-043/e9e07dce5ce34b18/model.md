#### Abstract Mathematical Model

**Index Sets:**
- $I$: Set of drug products (indexed by $i$), from products.csv.

**Parameters:**
- $v_i$: Value (benefit) per unit of drug $i$, from products.csv [Value].
- $w_i$: Weight (stock usage per unit) of drug $i$, from products.csv [Weight].
- $C$: Overall stock capacity, from capacity.csv [Capacity].

**Decision Variables:**
- $x_i$: Number of units of drug $i$ to order each day, $x_i \in \mathbb{Z}_+, \forall i \in I$.

**Objective:**
\[
\max \sum_{i \in I} v_i x_i
\]

**Constraints:**
1. **Overall Stock Capacity:**
   \[
   \sum_{i \in I} w_i x_i \leq C
   \]
2. **Non-negativity and Integrality:**
   \[
   x_i \in \mathbb{Z}_+, \quad \forall i \in I
   \]

---

#### Data Mapping

- **products.csv**
  - Table ID: file_1_view_0
  - Columns: ProductName (index set $I$), Value ($v_i$), Weight ($w_i$)
- **capacity.csv**
  - Table ID: file_0_view_0
  - Column: Capacity ($C$)