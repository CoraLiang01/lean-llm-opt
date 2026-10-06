**Mathematical Optimization Model**

**Sets:**
- \( S \): Set of distribution centers (sources), indexed by \( s \).  
  \( S = \{\text{S1}, \text{S2}, \ldots, \text{S12}\} \) from file_1_view_0."Unnamed: 0"
- \( C \): Set of customer groups (destinations), indexed by \( c \).  
  \( C = \{\text{C1}, \text{C2}, \ldots, \text{C12}\} \) from file_0_view_0."customer"

**Parameters:**
- \( d_c \): Demand of customer group \( c \).  
  From file_0_view_0:  
  \( d_{\text{C1}} = 52 \), \( d_{\text{C2}} = 80 \), ..., \( d_{\text{C12}} = 31 \)
- \( u_s \): Supply capacity of distribution center \( s \).  
  From file_1_view_0:  
  \( u_{\text{S1}} = 58 \), \( u_{\text{S2}} = 32 \), ..., \( u_{\text{S12}} = 948 \)
- \( t_{s,c} \): Transportation cost per unit from \( s \) to \( c \).  
  From file_2_view_0:  
  \( t_{s,c} = \) value in row \( s \), column \( c \)

**Decision Variables:**
- \( x_{s,c} \geq 0 \): Amount of goods shipped from distribution center \( s \) to customer group \( c \)

**Objective:**
\[
\min \sum_{s \in S} \sum_{c \in C} t_{s,c} \cdot x_{s,c}
\]

**Constraints:**

1. **Demand Satisfaction:**  
  Each customer group’s demand must be fully met:
\[
\forall c \in C: \quad \sum_{s \in S} x_{s,c} = d_c
\]

2. **Supply Capacity:**  
  No distribution center can ship more than its capacity:
\[
\forall s \in S: \quad \sum_{c \in C} x_{s,c} \leq u_s
\]

3. **Non-negativity:**  
\[
\forall s \in S,\, c \in C: \quad x_{s,c} \geq 0
\]

---

**Data Mapping**

- \( S \): file_1_view_0."Unnamed: 0"
- \( C \): file_0_view_0."customer"
- \( d_c \): file_0_view_0."demand" (row: customer \( c \))
- \( u_s \): file_1_view_0."supply_capacity" (row: distribution center \( s \))
- \( t_{s,c} \): file_2_view_0, row "Unnamed: 0" = \( s \), column \( c \)
- \( x_{s,c} \): Decision variable for all \( s \in S \), \( c \in C \)

---

**Complete Model (Symbolic):**

\[
\begin{align*}
\min_{x_{s,c} \geq 0} \quad & \sum_{s \in S} \sum_{c \in C} t_{s,c} \cdot x_{s,c} \\
\text{s.t.} \quad
& \sum_{s \in S} x_{s,c} = d_c \quad \forall c \in C \\
& \sum_{c \in C} x_{s,c} \leq u_s \quad \forall s \in S \\
& x_{s,c} \geq 0 \quad \forall s \in S,\, c \in C
\end{align*}
\]

**All indices, parameters, and coefficients are bound directly to the supplied data as specified above.**