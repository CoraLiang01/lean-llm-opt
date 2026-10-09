##### Mathematical Model

Let $I$ be the set of suppliers (from "supply_capacity.csv"): $I = \{\text{S1}, \text{S2}, \ldots, \text{S10}\}$.

Let $J$ be the set of customer groups (from "customer_demand.csv"): $J = \{\text{C1}, \text{C2}, \ldots, \text{C10}\}$.

Let $x_{ij} \geq 0$ be the continuous quantity shipped from supplier $i \in I$ to customer $j \in J$.

Parameters:
- $d_j$: demand of customer $j$ (from "customer_demand.csv")
- $s_i$: supply capacity of supplier $i$ (from "supply_capacity.csv")
- $c_{ij}$: transportation cost per unit from supplier $i$ to customer $j$ (from "transportation_costs.csv")

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

- $I$ (suppliers): all "supplier_id" in "supply_capacity.csv" and "transportation_costs.csv" (S1, S2, ..., S10)
- $J$ (customers): all "customer_id" in "customer_demand.csv" and columns "transportation_cost_to_C1", ..., "transportation_cost_to_C10" in "transportation_costs.csv" (C1, C2, ..., C10)
- $d_j$: "demand" column in "customer_demand.csv", indexed by "customer_id"
- $s_i$: "supply_capacity" column in "supply_capacity.csv", indexed by "supplier_id"
- $c_{ij}$: "transportation_cost_to_Ck" column in "transportation_costs.csv" for customer $j = \text{C}k$, row indexed by "supplier_id" $i$

Index sets, parameters, and all coefficients are taken directly from the current CSV files as described above.