##### Decision Variables

- $y_i \in \{0,1\}$: 1 if facility location $i$ is opened, 0 otherwise, for all $i \in I$.
- $x_{ij} \in \{0,1\}$: 1 if area $j$ is assigned to facility $i$, 0 otherwise, for all $i \in I$, $j \in J$.

##### Parameters

- $I$: set of candidate facility locations (from column "Location" in table_id file_1_view_0).
- $J$: set of residential areas (from column "Area" in table_id file_0_view_0).
- $d_j$: demand of area $j$ (from column "Demand" in table_id file_0_view_0).
- $c_{ij}$: distance from location $i$ to area $j$ (from column $j$ in table_id file_1_view_0, row $i$).
- $p$: number of facilities to open (from row where Parameter = "NumberOfFacilitiesToOpen" in table_id file_2_view_0, column "Value").

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} d_j \, c_{ij} \, x_{ij}
\]

##### Constraints

1. **Assignment:** Each area is assigned to exactly one facility:
   \[
   \sum_{i \in I} x_{ij} = 1 \quad \forall j \in J
   \]

2. **Facility count:** Exactly $p$ facilities are opened:
   \[
   \sum_{i \in I} y_i = p
   \]

3. **Assignment only to open facilities:** Areas can only be assigned to open facilities:
   \[
   x_{ij} \leq y_i \quad \forall i \in I,\, j \in J
   \]

4. **Binary restrictions:**
   \[
   x_{ij} \in \{0,1\} \quad \forall i \in I,\, j \in J
   \]
   \[
   y_i \in \{0,1\} \quad \forall i \in I
   \]

---

##### Data Mapping

- $I$: All values in column "Location" of table_id file_1_view_0.
- $J$: All values in column "Area" of table_id file_0_view_0.
- $d_j$: Value in column "Demand" for area $j$ in table_id file_0_view_0.
- $c_{ij}$: Value in column $j$ (area) and row $i$ (location) in table_id file_1_view_0.
- $p$: Value in column "Value" where "Parameter" = "NumberOfFacilitiesToOpen" in table_id file_2_view_0.