#### Index Sets

- $S$: set of Walmart stores (from supply_capacity.csv, table_id: file_1_view_0, column: Unnamed: 0)
- $C$: set of customer groups (from customer_demand.csv, table_id: file_0_view_0, column: customer)

#### Parameters

- $a_s$: supply capacity of store $s \in S$ (from supply_capacity.csv, table_id: file_1_view_0, column: supply_capacity)
- $d_c$: demand of customer group $c \in C$ (from customer_demand.csv, table_id: file_0_view_0, column: demand)
- $t_{s,c}$: transportation cost per unit from store $s$ to customer $c$ (from transportation_costs.csv, table_id: file_2_view_0, row: Unnamed: 0 = $s$, column: $c$)

#### Decision Variables

- $x_{s,c} \geq 0$: quantity transported from store $s$ to customer $c$, for all $s \in S$, $c \in C$

#### Objective

$$
\min \sum_{s \in S} \sum_{c \in C} t_{s,c} \cdot x_{s,c}
$$

#### Constraints

1. **Supply Capacity at Each Store:**
   $$
   \sum_{c \in C} x_{s,c} \leq a_s, \quad \forall s \in S
   $$

2. **Demand Satisfaction for Each Customer Group:**
   $$
   \sum_{s \in S} x_{s,c} = d_c, \quad \forall c \in C
   $$

3. **Nonnegativity:**
   $$
   x_{s,c} \geq 0, \quad \forall s \in S,\, c \in C
   $$

---

#### Data Mapping

- $S$ (stores): file_1_view_0, column "Unnamed: 0" (supply_capacity.csv)
- $a_s$: file_1_view_0, column "supply_capacity" (supply_capacity.csv)
- $C$ (customers): file_0_view_0, column "customer" (customer_demand.csv)
- $d_c$: file_0_view_0, column "demand" (customer_demand.csv)
- $t_{s,c}$: file_2_view_0, row "Unnamed: 0" = $s$, columns $C1$–$C12$ (transportation_costs.csv)

All records and columns from the three files are used as returned by CSVQA.