#### Index Sets

- $S$: set of production plants (indexed by $s$), from supply_capacity.csv [table_id: file_1_view_0, column: Unnamed: 0]
- $C$: set of retail outlets (indexed by $c$), from customer_demand.csv [table_id: file_0_view_0, column: customer]

#### Parameters

- $d_c$: daily demand at outlet $c \in C$  
  [from customer_demand.csv, table_id: file_0_view_0, column: demand]

- $u_s$: daily production capacity at plant $s \in S$  
  [from supply_capacity.csv, table_id: file_1_view_0, column: supply_capacity]

- $t_{s,c}$: transportation cost per unit from plant $s$ to outlet $c$  
  [from transportation_costs.csv, table_id: file_2_view_0, row: Unnamed: 0 (plant), columns: C1, C2, C3, C4 (outlets)]

#### Decision Variables

- $x_{s,c} \geq 0$: quantity of beverages shipped from plant $s$ to outlet $c$ (continuous, non-negative)

#### Objective

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
   x_{s,c} \geq 0, \quad \forall s \in S,\, c \in C
   $$

---

#### Data Mapping

- customer_demand.csv [table_id: file_0_view_0]:  
  customer (outlet identifier) $\to C$  
  demand $\to d_c$

- supply_capacity.csv [table_id: file_1_view_0]:  
  Unnamed: 0 (plant identifier) $\to S$  
  supply_capacity $\to u_s$

- transportation_costs.csv [table_id: file_2_view_0]:  
  Unnamed: 0 (plant identifier) $\to S$  
  C1, C2, C3, C4 (outlet columns) $\to t_{s,c}$ for each $s \in S$, $c \in C$

---

This abstract model uses all records and columns as returned by CSVQA, with index sets, parameters, variables, objective, and constraints defined symbolically and mapped to the exact data sources.