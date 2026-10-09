##### Decision Variables

$y_i \in \{0,1\}$: 1 if clinic $i$ is opened, 0 otherwise, for all clinics $i$ (from column "Clinic" in table_id file_1_view_0).

$x_{ij} \in \{0,1\}$: 1 if neighborhood $j$ (from column "Neighborhood" in table_id file_0_view_0) is assigned to clinic $i$, 0 otherwise.

##### Objective Function

$\min \sum_{i \in I} \sum_{j \in J} d_j \cdot c_{ij} \cdot x_{ij}$

where:
- $I$ = set of clinics (from "Clinic" in file_1_view_0)
- $J$ = set of neighborhoods (from "Neighborhood" in file_0_view_0)
- $d_j$ = demand of neighborhood $j$ (from "Demand" in file_0_view_0)
- $c_{ij}$ = distance from clinic $i$ to neighborhood $j$ (from column $j$ in file_1_view_0, row $i$)

##### Constraints

1. **Assignment:** Each neighborhood is assigned to exactly one clinic:
   $$
   \sum_{i \in I} x_{ij} = 1, \quad \forall j \in J
   $$

2. **Clinic opening:** Exactly $p$ clinics are opened:
   $$
   \sum_{i \in I} y_i = p
   $$
   where $p$ is the value in "Value" where "Parameter" = "NumberOfClinicsToOpen" in file_2_view_0.

3. **Assignment only to open clinics:** A neighborhood can only be assigned to an open clinic:
   $$
   x_{ij} \leq y_i, \quad \forall i \in I,\, j \in J
   $$

4. **Variable domains:**
   $$
   x_{ij} \in \{0,1\},\quad y_i \in \{0,1\}
   $$

---

##### Data Mapping

- Clinics $I$: column "Clinic" in table_id file_1_view_0
- Neighborhoods $J$: column "Neighborhood" in table_id file_0_view_0
- Demand $d_j$: column "Demand" in table_id file_0_view_0, indexed by "Neighborhood"
- Distance $c_{ij}$: entry in table_id file_1_view_0, row "Clinic" = $i$, column $j$ (neighborhood)
- Number of clinics to open $p$: "Value" where "Parameter" = "NumberOfClinicsToOpen" in table_id file_2_view_0

All indices, parameters, and matrix entries are to be taken exactly as defined in the source tables.