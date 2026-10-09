#### Mathematical Model

Let $S$ be the set of suppliers (from "supplier_id" in supply_capacity.csv and transportation_costs.csv), and $C$ the set of customer groups (from "customer_id" in customer_demand.csv and transportation_costs.csv).

Decision variables:
- $x_{ij} \geq 0$: quantity shipped from supplier $i \in S$ to customer $j \in C$ (continuous).

Objective:
\[
\min \sum_{i \in S} \sum_{j \in C} c_{ij} x_{ij}
\]
where $c_{ij}$ is the transportation cost per unit from supplier $i$ to customer $j$ (from "transportation_costs.csv").

Subject to:
- Demand satisfaction:
\[
\sum_{i \in S} x_{ij} \geq d_j \quad \forall j \in C
\]
where $d_j$ is the demand of customer $j$ (from "customer_demand.csv").

- Supply capacity:
\[
\sum_{j \in C} x_{ij} \leq s_i \quad \forall i \in S
\]
where $s_i$ is the supply capacity of supplier $i$ (from "supply_capacity.csv").

- Non-negativity:
\[
x_{ij} \geq 0 \quad \forall i \in S,\, j \in C
\]

#### Data Mapping

- $S$: All "supplier_id" in file_1_view_0 (supply_capacity.csv) and file_2_view_0 (transportation_costs.csv) rows.
- $C$: All "customer_id" in file_0_view_0 (customer_demand.csv) and columns "transportation_cost_to_Ck" in file_2_view_0 (transportation_costs.csv), $k=1,\ldots,10$.
- $d_j$: "demand" column in file_0_view_0, indexed by "customer_id".
- $s_i$: "supply_capacity" column in file_1_view_0, indexed by "supplier_id".
- $c_{ij}$: "transportation_cost_to_Ck" columns in file_2_view_0, with row "supplier_id" $i$ and column for customer $j$.

Index sets and all coefficients are defined by the current CSV records and columns as described above. No data is omitted or aggregated.