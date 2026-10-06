**Mathematical Optimization Model**

**Sets:**
- \( S \): Set of supply centers (distribution centers), indexed by \( s \).  
  \( S = \{\text{S1}, \text{S2}, \ldots, \text{S12}\} \) from file_1_view_0."Unnamed: 0"
- \( C \): Set of customer groups, indexed by \( c \).  
  \( C = \{\text{C1}, \text{C2}, \ldots, \text{C12}\} \) from file_0_view_0."customer"

**Parameters:**
- \( d_c \): Demand of customer group \( c \).  
  From file_0_view_0, column "demand", for each \( c \in C \).
- \( u_s \): Supply capacity of distribution center \( s \).  
  From file_1_view_0, column "supply_capacity", for each \( s \in S \).
- \( t_{s,c} \): Transportation cost per unit from supply center \( s \) to customer group \( c \).  
  From file_2_view_0, entry at row \( s \) ("Unnamed: 0") and column \( c \).

**Decision Variables:**
- \( x_{s,c} \geq 0 \): Number of units shipped from supply center \( s \) to customer group \( c \).

**Objective:**
\[
\min \sum_{s \in S} \sum_{c \in C} t_{s,c} \cdot x_{s,c}
\]

**Constraints:**

1. **Demand Satisfaction:**  
  For each customer group \( c \in C \):
\[
\sum_{s \in S} x_{s,c} = d_c
\]

2. **Supply Capacity:**  
  For each supply center \( s \in S \):
\[
\sum_{c \in C} x_{s,c} \leq u_s
\]

3. **Non-negativity:**  
  For all \( s \in S, c \in C \):
\[
x_{s,c} \geq 0
\]

---

**Data Mapping**

- \( S \): file_1_view_0."Unnamed: 0"
- \( C \): file_0_view_0."customer"
- \( d_c \): file_0_view_0."demand", indexed by "customer"
- \( u_s \): file_1_view_0."supply_capacity", indexed by "Unnamed: 0"
- \( t_{s,c} \): file_2_view_0, row "Unnamed: 0" = \( s \), column \( c \)

---

**Variable Domains:**
- \( x_{s,c} \in [0, \infty) \), continuous, for all \( s \in S, c \in C \)