#### Mathematical Model

Let $S$ be the set of suppliers (from column supplier_id in supply_capacity.csv and transportation_costs.csv), and $C$ the set of customer groups (from column customer_id in customer_demand.csv and transportation_costs.csv).

Let $x_{s,c} \geq 0$ be the continuous quantity shipped from supplier $s \in S$ to customer $c \in C$.

Parameters:
- $d_c$: demand of customer $c$ (from customer_demand.csv, column demand)
- $u_s$: supply capacity of supplier $s$ (from supply_capacity.csv, column supply_capacity)
- $c_{s,c}$: transportation cost per unit from supplier $s$ to customer $c$ (from transportation_costs.csv, column transportation_cost_to_Ck)

Objective:
\[
\min \sum_{s \in S} \sum_{c \in C} c_{s,c} \, x_{s,c}
\]

Subject to:
1. Demand satisfaction:
   \[
   \sum_{s \in S} x_{s,c} \geq d_c \qquad \forall c \in C
   \]
2. Supply capacity:
   \[
   \sum_{c \in C} x_{s,c} \leq u_s \qquad \forall s \in S
   \]
3. Nonnegativity:
   \[
   x_{s,c} \geq 0 \qquad \forall s \in S,\, c \in C
   \]

#### Data Mapping

- $S$: All supplier_id in supply_capacity.csv (table_id: file_1_view_0, column: supplier_id) and transportation_costs.csv (table_id: file_2_view_0, column: supplier_id)
- $C$: All customer_id in customer_demand.csv (table_id: file_0_view_0, column: customer_id) and as suffixes in transportation_costs.csv columns (table_id: file_2_view_0, columns: transportation_cost_to_Ck)
- $d_c$: For each $c$, value from customer_demand.csv (table_id: file_0_view_0, column: demand, key: customer_id)
- $u_s$: For each $s$, value from supply_capacity.csv (table_id: file_1_view_0, column: supply_capacity, key: supplier_id)
- $c_{s,c}$: For each $s,c$, value from transportation_costs.csv (table_id: file_2_view_0, row: supplier_id $s$, column: transportation_cost_to_Ck where $k$ matches $c$)

Index sets, parameters, and all coefficients are defined exactly as in the current CSV data. No data is omitted or aggregated. Variable domains and all constraints are as specified in the user query.