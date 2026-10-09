#### Mathematical Model

Let $I$ be the set of distribution centers (indexed by $i$), and $J$ the set of customer groups (indexed by $j$), as defined by the source data.

Let $x_{ij} \geq 0$ be the continuous quantity shipped from distribution center $i \in I$ to customer group $j \in J$.

Parameters:
- $d_j$: demand of customer group $j$ (from "customer_demand.csv")
- $s_i$: supply capacity of distribution center $i$ (from "supply_capacity.csv")
- $c_{ij}$: transportation cost per unit from distribution center $i$ to customer group $j$ (from "transportation_costs.csv")

Objective:
\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

Subject to:
1. Demand satisfaction:
   \[
   \sum_{i \in I} x_{ij} \geq d_j \quad \forall j \in J
   \]
2. Supply capacity:
   \[
   \sum_{j \in J} x_{ij} \leq s_i \quad \forall i \in I
   \]
3. Non-negativity:
   \[
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   \]

#### Data Mapping

- $I$: Set of distribution centers, from "supply_capacity.csv", column "Unnamed: 0", table_id "file_1_view_0"
- $J$: Set of customer groups, from "customer_demand.csv", column "customer", table_id "file_0_view_0"
- $d_j$: Demand for customer $j$, from "customer_demand.csv", column "demand", table_id "file_0_view_0"
- $s_i$: Supply capacity for distribution center $i$, from "supply_capacity.csv", column "supply_capacity", table_id "file_1_view_0"
- $c_{ij}$: Transportation cost per unit from $i$ to $j$, from "transportation_costs.csv", row id "Unnamed: 0" (distribution center), column $j$ (customer), table_id "file_2_view_0"
- $x_{ij}$: Decision variable, quantity shipped from $i$ to $j$ (continuous, nonnegative)

All index sets, parameters, and constraints are defined exactly as per the current source data and user query.