#### Abstract Mathematical Model

**Index Sets:**
- $S$: Set of distribution centers (suppliers), indexed by $s$.
- $D$: Set of customer groups (demands), indexed by $d$.

**Parameters:**
- $c_{sd}$: Per-unit transportation cost from supplier $s$ to customer group $d$.  
  (from table_id: file_2_view_0, columns: Unnamed: 0, demand1, ..., demand8)
- $a_s$: Supply capacity of supplier $s$.  
  (from table_id: file_1_view_0, columns: Supplier, supply_capacity)
- $b_d$: Demand of customer group $d$.  
  (from table_id: file_0_view_0, columns: Customers, demand)

**Decision Variables:**
- $x_{sd} \geq 0$: Quantity shipped from supplier $s$ to customer group $d$.

**Objective:**
\[
\min \sum_{s \in S} \sum_{d \in D} c_{sd} \, x_{sd}
\]

**Constraints:**
1. **Demand Satisfaction:**  
   $\displaystyle \sum_{s \in S} x_{sd} = b_d, \quad \forall d \in D$

2. **Supply Capacity:**  
   $\displaystyle \sum_{d \in D} x_{sd} \leq a_s, \quad \forall s \in S$

3. **Nonnegativity:**  
   $x_{sd} \geq 0, \quad \forall s \in S,\, d \in D$

---

**Data Mapping:**

- $S$ (suppliers): file_1_view_0, column "Supplier"; file_2_view_0, column "Unnamed: 0"
- $D$ (customer groups): file_0_view_0, column "Customers"; file_2_view_0, columns "demand1" through "demand8"
- $a_s$: file_1_view_0, column "supply_capacity"
- $b_d$: file_0_view_0, column "demand"
- $c_{sd}$: file_2_view_0, columns "Unnamed: 0" (supplier), "demand1"..."demand8" (customer group)