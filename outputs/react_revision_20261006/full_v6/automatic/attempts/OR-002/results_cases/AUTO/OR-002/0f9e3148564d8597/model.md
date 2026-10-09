##### Mathematical Model

Let $I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}, \text{S5}, \text{S6}, \text{S7}, \text{S8}, \text{S9}, \text{S10}, \text{S11}\}$ be the set of Walmart stores (suppliers), and $J = \{\text{C1}, \text{C2}, \text{C3}, \text{C4}, \text{C5}, \text{C6}, \text{C7}, \text{C8}, \text{C9}, \text{C10}, \text{C11}, \text{C12}\}$ the set of customer groups.

Let $x_{ij} \geq 0$ be the continuous quantity shipped from store $i \in I$ to customer group $j \in J$.

Parameters:
- $d_j$: demand of customer group $j$ (from customer_demand.csv)
- $s_i$: supply capacity of store $i$ (from supply_capacity.csv)
- $c_{ij}$: transportation cost per unit from store $i$ to customer group $j$ (from transportation_costs.csv)

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

##### Data Mapping

- $I$ (suppliers): All "Unnamed: 0" values in supply_capacity.csv (S1, S2, ..., S11), table_id: file_1_view_0
- $J$ (customers): All "customer" values in customer_demand.csv (C1, C2, ..., C12), table_id: file_0_view_0
- $d_j$: "demand" column in customer_demand.csv, indexed by "customer", table_id: file_0_view_0
- $s_i$: "supply_capacity" column in supply_capacity.csv, indexed by "Unnamed: 0", table_id: file_1_view_0
- $c_{ij}$: "C1"..."C12" columns in transportation_costs.csv, rows indexed by "Unnamed: 0" (supplier), table_id: file_2_view_0

Index sets, parameters, and cost matrix are defined by the full returned data; no entities are omitted or aggregated. Variable domains and all constraints are as specified in the user query.