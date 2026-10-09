##### Mathematical Model

Let $I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}, \text{S5}, \text{S6}, \text{S7}, \text{S8}, \text{S9}, \text{S10}\}$ be the set of suppliers, and $J = \{\text{C1}, \text{C2}, \text{C3}, \text{C4}, \text{C5}, \text{C6}, \text{C7}, \text{C8}, \text{C9}, \text{C10}\}$ the set of customer groups.

Let $x_{ij} \geq 0$ be the continuous quantity shipped from supplier $i \in I$ to customer $j \in J$.

Parameters:
- $d_j$: demand of customer $j \in J$
- $s_i$: supply capacity of supplier $i \in I$
- $c_{ij}$: transportation cost per unit from supplier $i$ to customer $j$

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

- $I$ (suppliers): all "Unnamed: 0" values from supply_capacity.csv and transportation_costs.csv rows, in source order.
- $J$ (customers): all "customer" values from customer_demand.csv and all columns (except "Unnamed: 0") from transportation_costs.csv, in source order.
- $d_j$: "demand" column from customer_demand.csv, indexed by "customer".
- $s_i$: "supply_capacity" column from supply_capacity.csv, indexed by "Unnamed: 0".
- $c_{ij}$: value at row $i$ ("Unnamed: 0" in transportation_costs.csv) and column $j$ (customer ID) in transportation_costs.csv.

All indices, parameters, and coefficients are to be taken exactly as returned in the Observation, preserving source order and identifiers.