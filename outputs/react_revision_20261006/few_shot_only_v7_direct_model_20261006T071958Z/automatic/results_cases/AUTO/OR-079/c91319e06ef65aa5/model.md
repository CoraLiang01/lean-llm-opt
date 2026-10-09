##### Decision Variables

- $y_i \in \{0,1\}$: 1 if factory $i \in I$ is constructed, 0 otherwise.
- $x_{ij} \geq 0$: quantity shipped from factory $i \in I$ to distribution center $j \in J$ (continuous).

##### Objective Function

\[
\min \sum_{i \in I} \text{FixedCost}_i \cdot y_i + \sum_{i \in I} \sum_{j \in J} \text{ShipCost}_{ij} \cdot x_{ij}
\]

##### Constraints

1. **Demand satisfaction at each distribution center:**
   \[
   \sum_{i \in I} x_{ij} = \text{Demand}_j, \quad \forall j \in J
   \]

2. **Factory capacity (if constructed):**
   \[
   \sum_{j \in J} x_{ij} \leq \text{Capacity}_i \cdot y_i, \quad \forall i \in I
   \]

3. **Variable domains:**
   \[
   y_i \in \{0,1\}, \quad \forall i \in I
   \]
   \[
   x_{ij} \geq 0, \quad \forall i \in I,\, j \in J
   \]

##### Index Sets

- $I$: Set of potential factories, from `Facility` in table_id `file_0_view_0`
- $J$: Set of distribution centers, from `Destination` in table_id `file_2_view_0`

##### Data Mapping

- $\text{FixedCost}_i$: `FixedCost` column for factory $i$ in table_id `file_0_view_0`
- $\text{Capacity}_i$: `Capacity` column for factory $i$ in table_id `file_0_view_0`
- $\text{ShipCost}_{ij}$: Entry in table_id `file_1_view_0`, row where `Origin` = $i$, column $j$
- $\text{Demand}_j$: `Demand` column for distribution center $j$ in table_id `file_2_view_0`