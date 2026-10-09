##### Decision Variables

- $y_i \in \{0,1\}$: 1 if facility $i$ is opened, 0 otherwise, for all $i \in I$.
- $x_{ij} \in \{0,1\}$: 1 if neighborhood $j$ is assigned to facility $i$, 0 otherwise, for all $i \in I$, $j \in J$.

##### Parameters

- $d_j$: demand of neighborhood $j$ (from column "Demand" in table_id file_0_view_0, key "Neighborhood").
- $c_{ij}$: distance from facility $i$ to neighborhood $j$ (from table_id file_1_view_0, row "Facility", column $j$).
- $p$: number of facilities to open (from table_id file_2_view_0, column "Value" where "Parameter" = "NumberOfFacilitiesToOpen").
- $I$: set of candidate facilities (from "Facility" in file_1_view_0).
- $J$: set of neighborhoods (from "Neighborhood" in file_0_view_0).

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} d_j \, c_{ij} \, x_{ij}
\]

##### Constraints

1. **Assignment:** Each neighborhood is assigned to exactly one facility:
   \[
   \sum_{i \in I} x_{ij} = 1 \quad \forall j \in J
   \]

2. **Facility Opening:** Exactly $p$ facilities are opened:
   \[
   \sum_{i \in I} y_i = p
   \]

3. **Assignment-to-Open-Facility Linking:** Neighborhoods can only be assigned to open facilities:
   \[
   x_{ij} \leq y_i \quad \forall i \in I,\, j \in J
   \]

4. **Binary Restrictions:**
   \[
   x_{ij} \in \{0,1\} \quad \forall i \in I,\, j \in J
   \]
   \[
   y_i \in \{0,1\} \quad \forall i \in I
   \]

---

##### Data Mapping

- $I$: All values in column "Facility" of table_id file_1_view_0.
- $J$: All values in column "Neighborhood" of table_id file_0_view_0.
- $d_j$: Column "Demand" in table_id file_0_view_0, keyed by "Neighborhood".
- $c_{ij}$: Entry in table_id file_1_view_0, row "Facility" = $i$, column $j$.
- $p$: Value in column "Value" of table_id file_2_view_0, where "Parameter" = "NumberOfFacilitiesToOpen".