#### Abstract Mathematical Optimization Model

**Index Sets:**
- $I$: Set of vehicle types (from products.csv, column ProductName).

**Parameters:**
- $v_i$: Benefit coefficient for vehicle type $i \in I$ (from products.csv, column Value).
- $w_i$: Inventory weight (or space requirement) for vehicle type $i \in I$ (from products.csv, column Weight).
- $C$: Total inventory capacity (from capacity.csv, column Capacity).

**Decision Variables:**
- $x_i$: Number of units of vehicle type $i \in I$ to order daily. ($x_i \in \mathbb{Z}_{\geq 0}$)

**Objective:**
\[
\max \sum_{i \in I} v_i x_i
\]

**Constraints:**
1. **Inventory Capacity Constraint:**
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
    - $I$: ProductName
    - $v_i$: Value
    - $w_i$: Weight
- **capacity.csv**
  - Table ID: file_0_view_0
    - $C$: Capacity

All parameters and sets are defined symbolically and mapped to their exact table and column sources. No literal data values or record counts are included.