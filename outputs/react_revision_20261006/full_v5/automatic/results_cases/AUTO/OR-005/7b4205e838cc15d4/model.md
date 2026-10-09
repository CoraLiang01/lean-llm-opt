##### Mathematical Model

Let $I = \{\text{supplier1}, \text{supplier2}, \text{supplier3}, \text{supplier4}, \text{supplier5}, \text{supplier6}, \text{supplier7}, \text{supplier8}\}$ be the set of distribution centers (suppliers), and $J = \{\text{demand1}, \text{demand2}, \text{demand3}, \text{demand4}, \text{demand5}, \text{demand6}, \text{demand7}, \text{demand8}\}$ be the set of customer groups.

Let $x_{ij} \geq 0$ be the continuous quantity shipped from supplier $i \in I$ to customer $j \in J$.

Parameters:
- $d_j$: demand of customer $j$ (from customer_demand.csv)
- $s_i$: supply capacity of supplier $i$ (from supply_capacity.csv)
- $c_{ij}$: transportation cost per unit from supplier $i$ to customer $j$ (from transportation_costs.csv)

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

- $I$ (suppliers): All "Supplier" values in supply_capacity.csv, in source order.
- $J$ (customers): All "Customers" values in customer_demand.csv, in source order.
- $d_j$: "demand" column in customer_demand.csv, indexed by "Customers".
- $s_i$: "supply_capacity" column in supply_capacity.csv, indexed by "Supplier".
- $c_{ij}$: Entry in transportation_costs.csv at row with "Unnamed: 0" = supply$k$ (mapped to supplier$k$), column = demand$m$ (mapped to demand$m$), for all $i \in I$, $j \in J$.

Index and parameter values are to be taken exactly as listed in the source files, preserving all identifiers and coefficients.