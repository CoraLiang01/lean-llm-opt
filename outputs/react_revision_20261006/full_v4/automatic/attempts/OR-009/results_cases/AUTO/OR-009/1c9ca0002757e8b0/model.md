##### Decision Variables

Let $x_{ij} \geq 0$ be the quantity of beverages shipped from production plant $i$ to retail outlet $j$, for all $i \in I$, $j \in J$.

##### Index Sets

- $I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}\}$ (production plants, from supply_capacity.csv)
- $J = \{\text{C1}, \text{C2}, \text{C3}, \text{C4}\}$ (retail outlets, from customer_demand.csv)

##### Parameters

- $d_j$: demand of retail outlet $j$ (from customer_demand.csv)
- $s_i$: supply capacity of plant $i$ (from supply_capacity.csv)
- $c_{ij}$: transportation cost per unit from plant $i$ to outlet $j$ (from transportation_costs.csv)

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

##### Constraints

1. **Demand satisfaction:**  
   For each retail outlet $j \in J$,
   \[
   \sum_{i \in I} x_{ij} \geq d_j
   \]

2. **Supply capacity:**  
   For each plant $i \in I$,
   \[
   \sum_{j \in J} x_{ij} \leq s_i
   \]

3. **Non-negativity:**  
   \[
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   \]

##### Data Mapping

- $I$ (plants): S1, S2, S3, S4 (file_1_view_0, column "Unnamed: 0")
- $J$ (retail outlets): C1, C2, C3, C4 (file_0_view_0, column "customer")
- $d_j$: file_0_view_0, column "demand", indexed by "customer"
- $s_i$: file_1_view_0, column "supply_capacity", indexed by "Unnamed: 0"
- $c_{ij}$: file_2_view_0, columns "C1", "C2", "C3", "C4", rows indexed by "Unnamed: 0" (plants S1–S4)

No data omitted or invented; all identifiers and coefficients are preserved as in the source.