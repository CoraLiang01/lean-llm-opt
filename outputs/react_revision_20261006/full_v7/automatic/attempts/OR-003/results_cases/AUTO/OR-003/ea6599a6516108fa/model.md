##### Mathematical Model

Let $I = \{\text{S1}, \text{S2}, \ldots, \text{S10}\}$ be the set of suppliers, and $J = \{\text{C1}, \text{C2}, \ldots, \text{C10}\}$ be the set of customer groups.

Let $x_{ij} \geq 0$ denote the quantity shipped from supplier $i \in I$ to customer $j \in J$ (continuous).

Parameters:
- $d_j$: demand of customer $j \in J$ (from "customer_demand.csv")
- $s_i$: supply capacity of supplier $i \in I$ (from "supply_capacity.csv")
- $c_{ij}$: transportation cost per unit from supplier $i$ to customer $j$ (from "transportation_costs.csv")

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

- $I$ (suppliers): "Unnamed: 0" column in "supply_capacity.csv" and "transportation_costs.csv" (row labels)
- $J$ (customers): "customer" column in "customer_demand.csv" and "transportation_costs.csv" (column labels)
- $d_j$: "demand" column in "customer_demand.csv", indexed by "customer"
- $s_i$: "supply_capacity" column in "supply_capacity.csv", indexed by "Unnamed: 0"
- $c_{ij}$: "transportation_costs.csv", entry at row "Unnamed: 0" = $i$, column $j$ (e.g., "C1", ..., "C10")

All indices, parameters, and coefficients are to be taken exactly as listed in the respective CSV files.