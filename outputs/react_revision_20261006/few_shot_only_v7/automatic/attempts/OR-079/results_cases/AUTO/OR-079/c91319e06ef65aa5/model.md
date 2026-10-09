##### Decision Variables

- $y_i \in \{0,1\}$: 1 if factory $i \in I$ is constructed, 0 otherwise.
- $x_{ij} \geq 0$: quantity shipped from factory $i \in I$ to distribution center $j \in J$ (continuous).

##### Parameters

- $I = \{\text{A1}, \text{A2}, \ldots, \text{A15}\}$: set of potential factories.
- $J = \{\text{B1}, \text{B2}, \ldots, \text{B8}\}$: set of distribution centers.
- $f_i$: fixed cost of constructing factory $i$ (from facility_costs.csv).
- $u_i$: capacity of factory $i$ (from facility_costs.csv).
- $d_j$: demand at distribution center $j$ (from demand_requirements.csv).
- $c_{ij}$: unit shipping cost from factory $i$ to distribution center $j$ (from shipping_costs.csv).

##### Objective Function

\[
\min \sum_{i \in I} f_i y_i + \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

##### Constraints

1. **Demand satisfaction:**  
   \[
   \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J
   \]

2. **Factory capacity:**  
   \[
   \sum_{j \in J} x_{ij} \leq u_i y_i, \quad \forall i \in I
   \]

3. **Variable domains:**  
   \[
   x_{ij} \geq 0, \quad \forall i \in I,\, j \in J
   \]
   \[
   y_i \in \{0,1\}, \quad \forall i \in I
   \]

---

#### Data Mapping

- $I$ (factories): A1, A2, ..., A15 (facility_costs.csv, column "Facility")
- $J$ (distribution centers): B1, B2, ..., B8 (demand_requirements.csv, column "Destination"; shipping_costs.csv, columns "B1"-"B8")
- $f_i$: facility_costs.csv, column "FixedCost"
- $u_i$: facility_costs.csv, column "Capacity"
- $d_j$: demand_requirements.csv, column "Demand"
- $c_{ij}$: shipping_costs.csv, row "Origin" = $i$, column $j$ ("B1"-"B8")

All parameters are to be taken directly from the respective columns and rows of the provided CSV files, preserving all identifiers and values.