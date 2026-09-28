#### Abstract Mathematical Model

**Index Sets:**
- $I$: Set of vehicle types (indexed by $i$).

**Parameters:**
- $v_i$: Profit (benefit) per unit of vehicle type $i$.  
  [Source: products.csv, column "Value"]
- $w_i$: Inventory weight (stock usage) per unit of vehicle type $i$.  
  [Source: products.csv, column "Weight"]
- $C$: Total inventory capacity.  
  [Source: capacity.csv, column "Capacity"]

**Decision Variables:**
- $x_i$: Number of vehicles of type $i$ to order per day.  
  Domain: $x_i \in \mathbb{Z}_{\geq 0}$ (non-negative integers), $\forall i \in I$

**Objective:**
\[
\max \sum_{i \in I} v_i \cdot x_i
\]

**Constraints:**
1. **Inventory Capacity Constraint:**
   \[
   \sum_{i \in I} w_i \cdot x_i \leq C
   \]
2. **Non-negativity and Integrality:**
   \[
   x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i \in I
   \]

---

#### Data Mapping

- **products.csv**
  - "ProductName": Vehicle type identifier ($i \in I$)
  - "Value": Profit per vehicle ($v_i$)
  - "Weight": Inventory weight per vehicle ($w_i$)
- **capacity.csv**
  - "Capacity": Total inventory capacity ($C$)