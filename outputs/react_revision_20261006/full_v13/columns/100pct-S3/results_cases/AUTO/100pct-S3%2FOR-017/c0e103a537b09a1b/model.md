#### Mathematical Model

Let $I$ be the set of suppliers (from the supplier_id column of supply_capacity.csv and transportation_costs.csv), and $J$ be the set of customer groups (from the customer_id column of customer_demand.csv and the transportation_costs.csv columns).

Let $x_{ij} \geq 0$ be the continuous quantity shipped from supplier $i \in I$ to customer $j \in J$.

Parameters:
- $d_j$: demand of customer $j$ (from customer_demand.csv)
- $s_i$: supply capacity of supplier $i$ (from supply_capacity.csv)
- $c_{ij}$: transportation cost per unit from supplier $i$ to customer $j$ (from transportation_costs.csv)

Objective:
\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

Subject to:
\[
\sum_{i \in I} x_{ij} \geq d_j \qquad \forall j \in J
\]
\[
\sum_{j \in J} x_{ij} \leq s_i \qquad \forall i \in I
\]
\[
x_{ij} \geq 0 \qquad \forall i \in I,\, j \in J
\]

#### Data Mapping

- $I$ (suppliers): All supplier_id in supply_capacity.csv (file_1_view_0, column supplier_id) and transportation_costs.csv (file_2_view_0, column supplier_id)
- $J$ (customers): All customer_id in customer_demand.csv (file_0_view_0, column customer_id) and all columns with suffix transportation_cost_to_* in transportation_costs.csv (file_2_view_0)
- $d_j$: For each $j \in J$, demand from file_0_view_0, column demand, row where customer_id = $j$
- $s_i$: For each $i \in I$, supply_capacity from file_1_view_0, column supply_capacity, row where supplier_id = $i$
- $c_{ij}$: For each $i \in I$, $j \in J$, value in file_2_view_0, row where supplier_id = $i$, column transportation_cost_to_$j$

Index sets, parameters, and all coefficients are defined exactly as above, using the current source data. No data is omitted or aggregated. Variable domains and all constraints are as specified in the user query.