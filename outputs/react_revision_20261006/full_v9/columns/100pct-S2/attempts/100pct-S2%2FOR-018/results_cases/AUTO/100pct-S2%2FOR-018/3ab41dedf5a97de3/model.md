Mathematical Model

Let $I$ be the set of distribution centers (suppliers) and $J$ the set of customer groups, as defined by the identifiers in the source data.

Decision Variables:
$$
x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
$$
where $x_{ij}$ is the quantity shipped from distribution center $i$ to customer group $j$.

Parameters:
- $d_j$: demand of customer group $j$ (from "customer_demand.csv")
- $s_i$: supply capacity of distribution center $i$ (from "supply_capacity.csv")
- $c_{ij}$: transportation cost per unit from $i$ to $j$ (from "transportation_costs.csv")

Objective:
$$
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
$$

Subject to:
1. Demand satisfaction:
$$
\sum_{i \in I} x_{ij} \geq d_j \quad \forall j \in J
$$

2. Supply capacity:
$$
\sum_{j \in J} x_{ij} \leq s_i \quad \forall i \in I
$$

3. Non-negativity:
$$
x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
$$

Index sets and parameter mapping:

- $I = \{$S1, S2, S3, S4, S5, S6, S7, S8, S9, S10, S11, S12$\}$ (from "supplier_id" in "supply_capacity.csv" and "transportation_costs.csv")
- $J = \{$C1, C2, C3, C4, C5, C6, C7, C8, C9, C10, C11, C12$\}$ (from "customer_id" in "customer_demand.csv" and columns in "transportation_costs.csv")

Data Mapping

- $d_j$: "customer_demand.csv", table_id: file_0_view_0, column: "demand", key: "customer_id"
- $s_i$: "supply_capacity.csv", table_id: file_1_view_0, column: "supply_capacity", key: "supplier_id"
- $c_{ij}$: "transportation_costs.csv", table_id: file_2_view_0, row key: "supplier_id", column key: "transportation_cost_to_{j}" (where $j$ matches "customer_id" in $J$)

Variable domain and all constraints are as specified in the user query and mapped to the current data. No data is omitted or aggregated.