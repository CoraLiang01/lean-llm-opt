##### Symbolic Model

Let $I = \{S1, S2, \ldots, S18\}$ (distribution centers, as in supply_capacity.csv, in source order)  
Let $J = \{C1, C2, \ldots, C18\}$ (customer groups, as in customer_demand.csv, in source order)  
Let $x_{ij} \geq 0$ be the continuous quantity shipped from distribution center $i \in I$ to customer group $j \in J$.

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

Where:
- $d_j$ is the demand for customer group $j$ (from customer_demand.csv)
- $s_i$ is the supply capacity of distribution center $i$ (from supply_capacity.csv)
- $c_{ij}$ is the transportation cost per unit from $i$ to $j$ (from transportation_costs.csv, with rows indexed by $i$ and columns by $j$ in source order)

---

##### Data Mapping

- $I$ (distribution centers): All values in column "Unnamed: 0" of supply_capacity.csv (table_id: file_1_view_0), source order S1–S18.
- $J$ (customer groups): All values in column "customer" of customer_demand.csv (table_id: file_0_view_0), source order C1–C18.
- $d_j$: For each $j \in J$, value in column "demand" of customer_demand.csv (table_id: file_0_view_0), source order.
- $s_i$: For each $i \in I$, value in column "supply_capacity" of supply_capacity.csv (table_id: file_1_view_0), source order.
- $c_{ij}$: For each $i \in I$, $j \in J$, value in transportation_costs.csv (table_id: file_2_view_0), row "Unnamed: 0" = $i$, column $j$.

All indices, coefficients, and constraints are bound exactly to the retrieved data in source order. No data is omitted or aggregated.