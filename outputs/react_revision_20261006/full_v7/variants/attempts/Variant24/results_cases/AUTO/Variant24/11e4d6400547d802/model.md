##### Decision Variables

- $y_i \in \{0,1\}$: 1 if facility $i$ is opened, 0 otherwise, for all $i \in I$.
- $x_{ij} \in \{0,1\}$: 1 if neighborhood $j$ is assigned to facility $i$, 0 otherwise, for all $i \in I$, $j \in J$.

##### Parameters

- $I$: set of candidate facilities (Facility column in table_id=file_1_view_0).
- $J$: set of neighborhoods (Neighborhood column in table_id=file_0_view_0).
- $d_j$: demand of neighborhood $j$ (Demand column in table_id=file_0_view_0).
- $c_{ij}$: distance from facility $i$ to neighborhood $j$ (entry for Facility $i$, column $j$ in table_id=file_1_view_0).
- $p$: number of facilities to open (Value where Parameter = NumberOfFacilitiesToOpen in table_id=file_2_view_0).

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} d_j \, c_{ij} \, x_{ij}
\]

##### Constraints

1. **Assignment:** Each neighborhood is assigned to exactly one facility:
   \[
   \sum_{i \in I} x_{ij} = 1, \quad \forall j \in J
   \]
2. **Facility Opening:** Exactly $p$ facilities are opened:
   \[
   \sum_{i \in I} y_i = p
   \]
3. **Assignment-to-Open-Facility Linking:** Neighborhoods can only be assigned to open facilities:
   \[
   x_{ij} \leq y_i, \quad \forall i \in I,\, j \in J
   \]
4. **Variable Domains:**
   \[
   x_{ij} \in \{0,1\}, \quad y_i \in \{0,1\}
   \]

---

##### Data Mapping

- $I$: Facility values from column "Facility" in table_id=file_1_view_0.
- $J$: Neighborhood values from column "Neighborhood" in table_id=file_0_view_0.
- $d_j$: Demand from column "Demand" in table_id=file_0_view_0, indexed by "Neighborhood".
- $c_{ij}$: Distance from row "Facility" $i$, column $j$ in table_id=file_1_view_0.
- $p$: Value from column "Value" where "Parameter" = "NumberOfFacilitiesToOpen" in table_id=file_2_view_0.