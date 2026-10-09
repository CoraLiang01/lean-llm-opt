#### Index Sets

- $S$: set of production plants (indexed by $s$), from column "Unnamed: 0" in supply_capacity.csv and transportation_costs.csv.
- $C$: set of retail outlets (indexed by $c$), from column "customer" in customer_demand.csv and columns "C1", "C2", ... in transportation_costs.csv.

#### Parameters

- $d_c$: daily demand at outlet $c$, from column "demand" in customer_demand.csv.
- $u_s$: daily supply capacity at plant $s$, from column "supply_capacity" in supply_capacity.csv.
- $t_{s,c}$: transportation cost per unit from plant $s$ to outlet $c$, from table transportation_costs.csv, row "Unnamed: 0" = $s$, column $c$.

#### Decision Variables

- $x_{s,c} \geq 0$: quantity of beverages shipped from plant $s$ to outlet $c$ (continuous, non-negative).

#### Objective

Minimize total transportation cost:
$$
\min \sum_{s \in S} \sum_{c \in C} t_{s,c} \cdot x_{s,c}
$$

#### Constraints

1. **Demand Satisfaction at Each Outlet:**
   $$
   \sum_{s \in S} x_{s,c} = d_c, \quad \forall c \in C
   $$

2. **Supply Capacity at Each Plant:**
   $$
   \sum_{c \in C} x_{s,c} \leq u_s, \quad \forall s \in S
   $$

3. **Non-negativity:**
   $$
   x_{s,c} \geq 0, \quad \forall s \in S, \forall c \in C
   $$

---

#### Data Mapping

- **customer_demand.csv**:  
  - Table ID: file_0_view_0  
  - Columns: "customer" ($C$), "demand" ($d_c$)

- **supply_capacity.csv**:  
  - Table ID: file_1_view_0  
  - Columns: "Unnamed: 0" ($S$), "supply_capacity" ($u_s$)

- **transportation_costs.csv**:  
  - Table ID: file_2_view_0  
  - Rows: "Unnamed: 0" ($S$)  
  - Columns: $C$ (e.g., "C1", "C2", ...)  
  - Values: $t_{s,c}$

All records from each file are included as per the query. No additional filtering was applied.