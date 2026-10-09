##### Decision Variables

$y_i \in \{0,1\}$: 1 if facility $i$ is opened, 0 otherwise, for all $i \in I$.

$x_{ij} \in \{0,1\}$: 1 if neighborhood $j$ is assigned to facility $i$, 0 otherwise, for all $i \in I$, $j \in J$.

##### Parameters

- $I$: set of candidate facilities (Facility column in file_1_view_0)
- $J$: set of neighborhoods (Neighborhood column in file_0_view_0)
- $d_j$: demand of neighborhood $j$ (Demand column in file_0_view_0)
- $c_{ij}$: distance from facility $i$ to neighborhood $j$ (entry for row $i$, column $j$ in file_1_view_0)
- $p$: number of facilities to open (Value where Parameter = NumberOfFacilitiesToOpen in file_2_view_0)

##### Objective Function

$\min \sum_{i \in I} \sum_{j \in J} d_j \, c_{ij} \, x_{ij}$

##### Constraints

1. **Assignment:** Each neighborhood is assigned to exactly one facility:
   $$
   \sum_{i \in I} x_{ij} = 1, \quad \forall j \in J
   $$

2. **Facility Opening:** Exactly $p$ facilities are opened:
   $$
   \sum_{i \in I} y_i = p
   $$

3. **Assignment-to-Open-Facility Linking:** Neighborhoods can only be assigned to open facilities:
   $$
   x_{ij} \leq y_i, \quad \forall i \in I,\, j \in J
   $$

4. **Binary Restrictions:**
   $$
   x_{ij} \in \{0,1\}, \quad \forall i \in I,\, j \in J
   $$
   $$
   y_i \in \{0,1\}, \quad \forall i \in I
   $$

---

##### Data Mapping

- $I$: Facility (file_1_view_0, column "Facility")
- $J$: Neighborhood (file_0_view_0, column "Neighborhood")
- $d_j$: Demand (file_0_view_0, column "Demand", indexed by "Neighborhood")
- $c_{ij}$: facility-neighborhood distance (file_1_view_0, row "Facility", column "Neighborhood")
- $p$: Value (file_2_view_0, row where Parameter = "NumberOfFacilitiesToOpen", column "Value")