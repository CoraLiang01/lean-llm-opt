#### Abstract Mathematical Optimization Model

**Index Sets:**
- $S$: Set of distribution centers (from supply_capacity.csv, column "Unnamed: 0")
- $C$: Set of customer groups (from customer_demand.csv, column "customer")

**Parameters:**
- $d_c$: Demand of customer group $c \in C$ (from customer_demand.csv, column "demand")
- $u_s$: Supply capacity of distribution center $s \in S$ (from supply_capacity.csv, column "supply_capacity")
- $t_{s,c}$: Transportation cost per unit from distribution center $s \in S$ to customer group $c \in C$ (from transportation_costs.csv, columns "Unnamed: 0" for $s$ and $C1,\ldots,C12$ for $c$)

**Decision Variables:**
- $x_{s,c} \geq 0$: Number of units shipped from distribution center $s$ to customer group $c$ (continuous, non-negative)

**Objective:**
\[
\min \sum_{s \in S} \sum_{c \in C} t_{s,c} \cdot x_{s,c}
\]

**Constraints:**

1. **Demand Satisfaction:**
   \[
   \sum_{s \in S} x_{s,c} = d_c, \quad \forall c \in C
   \]

2. **Supply Capacity:**
   \[
   \sum_{c \in C} x_{s,c} \leq u_s, \quad \forall s \in S
   \]

3. **Non-negativity:**
   \[
   x_{s,c} \geq 0, \quad \forall s \in S,\, c \in C
   \]

---

#### Data Mapping

- **customer_demand.csv**: 
  - Table ID: file_0_view_0
  - Column "customer" $\rightarrow$ $C$
  - Column "demand" $\rightarrow$ $d_c$
- **supply_capacity.csv**: 
  - Table ID: file_1_view_0
  - Column "Unnamed: 0" $\rightarrow$ $S$
  - Column "supply_capacity" $\rightarrow$ $u_s$
- **transportation_costs.csv**: 
  - Table ID: file_2_view_0
  - Row "Unnamed: 0" $\rightarrow$ $s \in S$
  - Columns "C1", ..., "C12" $\rightarrow$ $c \in C$
  - Cell value $\rightarrow$ $t_{s,c}$

All index sets, parameters, and relationships are defined exactly as in the source tables and columns.