#### Abstract Mathematical Model

**Index Sets:**
- $I$: Set of vehicle types (indexed by $i$).

**Parameters:**
- $v_i$: Profit (benefit) from selling one unit of vehicle type $i$.
- $w_i$: Inventory weight (stock usage) of one unit of vehicle type $i$.
- $C$: Overall inventory capacity.

**Decision Variables:**
- $x_i \in \mathbb{Z}_{\geq 0}$: Number of vehicles of type $i$ to order per day.

**Objective:**
\[
\max \sum_{i \in I} v_i x_i
\]

**Constraints:**
1. **Overall Inventory Capacity:**
   \[
   \sum_{i \in I} w_i x_i \leq C
   \]
2. **Non-negativity and Integrality:**
   \[
   x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i \in I
   \]

---

#### Data Mapping

- **products.csv** (`file_1_view_0`):
  - $I$: All records in column `ProductName`
  - $v_i$: Column `Value`
  - $w_i$: Column `Weight`
- **capacity.csv** (`file_0_view_0`):
  - $C$: Column `Capacity` (single record)

All records from both files are used as returned by the query. No additional filtering or aggregation is applied.