##### Decision Variables

Let $x_{ij} \geq 0$ be the continuous quantity shipped from distribution center $i$ to customer group $j$.

- $i \in I$ where $I = \{$S1, S2, S3, S4, S5, S6, S7, S8, S9, S10, S11, S12$\}$ (from "supply_capacity.csv", column "Unnamed: 0")
- $j \in J$ where $J = \{$C1, C2, C3, C4, C5, C6, C7, C8, C9, C10, C11, C12$\}$ (from "customer_demand.csv", column "customer")

##### Parameters

- $d_j$: demand of customer group $j$ (from "customer_demand.csv", column "demand")
- $s_i$: supply capacity of distribution center $i$ (from "supply_capacity.csv", column "supply_capacity")
- $c_{ij}$: transportation cost per unit from $i$ to $j$ (from "transportation_costs.csv", row $i$, column $j$)

##### Objective

Minimize total transportation cost:
$$
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
$$

##### Constraints

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

- $I$ (distribution centers): "supply_capacity.csv", column "Unnamed: 0", table_id: file_1_view_0
- $J$ (customer groups): "customer_demand.csv", column "customer", table_id: file_0_view_0
- $d_j$: "customer_demand.csv", column "demand", table_id: file_0_view_0, indexed by $j$
- $s_i$: "supply_capacity.csv", column "supply_capacity", table_id: file_1_view_0, indexed by $i$
- $c_{ij}$: "transportation_costs.csv", row "Unnamed: 0" = $i$, column $j$, table_id: file_2_view_0

Index sets, parameters, and cost matrix are to be taken exactly as defined in the returned tables, preserving all identifiers and source order.