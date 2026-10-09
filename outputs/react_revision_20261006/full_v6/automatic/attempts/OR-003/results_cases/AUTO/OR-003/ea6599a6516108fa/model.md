##### Mathematical Model

Let $I = \{\text{S1}, \text{S2}, \ldots, \text{S10}\}$ be the set of suppliers, and $J = \{\text{C1}, \text{C2}, \ldots, \text{C10}\}$ the set of customer groups.

Let $x_{ij} \geq 0$ be the continuous quantity shipped from supplier $i \in I$ to customer group $j \in J$.

Parameters:
- $d_j$: demand of customer group $j$ (from "customer_demand.csv")
- $s_i$: supply capacity of supplier $i$ (from "supply_capacity.csv")
- $c_{ij}$: transportation cost per unit from supplier $i$ to customer group $j$ (from "transportation_costs.csv")

Objective:
\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

Subject to:
\[
\sum_{i \in I} x_{ij} \geq d_j \quad \forall j \in J
\]
\[
\sum_{j \in J} x_{ij} \leq s_i \quad \forall i \in I
\]
\[
x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
\]

##### Data Mapping

- $I$ (suppliers): all "Unnamed: 0" values in "supply_capacity.csv" and "transportation_costs.csv" (S1, ..., S10)
- $J$ (customers): all "customer" values in "customer_demand.csv" and columns C1, ..., C10 in "transportation_costs.csv"
- $d_j$: "demand" column in "customer_demand.csv", indexed by "customer"
- $s_i$: "supply_capacity" column in "supply_capacity.csv", indexed by "Unnamed: 0"
- $c_{ij}$: value in "transportation_costs.csv" at row "Unnamed: 0" = $i$, column $j$ (C1, ..., C10)

All indices, parameters, and coefficients are to be taken exactly as listed in the retrieved tables.