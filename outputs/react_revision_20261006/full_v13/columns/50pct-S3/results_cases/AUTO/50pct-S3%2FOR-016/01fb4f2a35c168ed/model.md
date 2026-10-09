##### Abstract Symbolic Mathematical Model

Let:
- $I$ = set of distribution centers (suppliers), indexed by $i$, from the "supplier_id" column of supply_capacity.csv and transportation_costs.csv.
- $J$ = set of customer groups, indexed by $j$, from the "customer_id" column of customer_demand.csv and the transportation_costs.csv columns (suffix after "transportation_cost_to_").
- $d_j$ = demand of customer group $j$, from "demand_units" in customer_demand.csv.
- $s_i$ = supply capacity of distribution center $i$, from "supply_capacity_units" in supply_capacity.csv.
- $c_{ij}$ = transportation cost per unit from distribution center $i$ to customer group $j$, from "transportation_cost_to_C*" columns in transportation_costs.csv.

Decision variables:
- $x_{ij} \geq 0$: quantity shipped from distribution center $i$ to customer group $j$ (continuous).

Objective:
$$
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
$$

Subject to:
1. Demand satisfaction for each customer group:
$$
\sum_{i \in I} x_{ij} \geq d_j \qquad \forall j \in J
$$

2. Supply capacity for each distribution center:
$$
\sum_{j \in J} x_{ij} \leq s_i \qquad \forall i \in I
$$

3. Non-negativity:
$$
x_{ij} \geq 0 \qquad \forall i \in I,\, j \in J
$$

##### Data Mapping

- $I$ (distribution centers): All "supplier_id" values in supply_capacity.csv (file_1_view_0) and transportation_costs.csv (file_2_view_0) rows.
- $J$ (customer groups): All "customer_id" values in customer_demand.csv (file_0_view_0) and all columns "transportation_cost_to_C*" in transportation_costs.csv (file_2_view_0), mapped to $j$ by suffix.
- $d_j$: "demand_units" in customer_demand.csv (file_0_view_0), indexed by "customer_id".
- $s_i$: "supply_capacity_units" in supply_capacity.csv (file_1_view_0), indexed by "supplier_id".
- $c_{ij}$: "transportation_cost_to_C*" columns in transportation_costs.csv (file_2_view_0), with row "supplier_id" = $i$ and column suffix $j$.

All index sets and parameters are defined by the full set of current records in the respective files, preserving source order and identifiers. No data is omitted or aggregated. Variable domains and all constraints are as specified in the user query.