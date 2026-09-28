#### Abstract Mathematical Model

**Index Sets:**
- $S$: Set of production plants (indexed by $s$), from supply_capacity.csv [file_1_view_0, Unnamed: 0]
- $C$: Set of retail outlets (indexed by $c$), from customer_demand.csv [file_0_view_0, customer]

**Parameters:**
- $d_c$: Demand at retail outlet $c \in C$, from customer_demand.csv [file_0_view_0, demand]
- $u_s$: Supply capacity at plant $s \in S$, from supply_capacity.csv [file_1_view_0, supply_capacity]
- $t_{s,c}$: Transportation cost per unit from plant $s$ to outlet $c$, from transportation_costs.csv [file_2_view_0, (row: Unnamed: 0 = $s$, column: $c$)]

**Decision Variables:**
- $x_{s,c} \geq 0$: Quantity of beverages shipped from plant $s$ to outlet $c$ (continuous, non-negative)

**Objective:**
\[
\min \sum_{s \in S} \sum_{c \in C} t_{s,c} \cdot x_{s,c}
\]

**Constraints:**

1. **Demand Satisfaction at Each Outlet:**
   \[
   \sum_{s \in S} x_{s,c} = d_c, \quad \forall c \in C
   \]

2. **Supply Capacity at Each Plant:**
   \[
   \sum_{c \in C} x_{s,c} \leq u_s, \quad \forall s \in S
   \]

3. **Non-negativity:**
   \[
   x_{s,c} \geq 0, \quad \forall s \in S,\, c \in C
   \]

---

#### Data Mapping

- **customer_demand.csv** [table_id: file_0_view_0]
  - Index set $C$ from column "customer"
  - Parameter $d_c$ from column "demand"
- **supply_capacity.csv** [table_id: file_1_view_0]
  - Index set $S$ from column "Unnamed: 0"
  - Parameter $u_s$ from column "supply_capacity"
- **transportation_costs.csv** [table_id: file_2_view_0]
  - Parameter $t_{s,c}$ from row "Unnamed: 0" = $s$, column $c$ (matrix over $S \times C$)

---

This model determines the optimal shipment plan from each plant to each outlet to minimize total transportation cost, while meeting all outlet demands and not exceeding any plant's capacity.