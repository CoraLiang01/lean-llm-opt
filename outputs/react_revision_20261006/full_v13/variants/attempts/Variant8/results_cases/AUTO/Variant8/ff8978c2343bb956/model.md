##### Decision Variables

- $y_i \in \{0,1\}$: 1 if clinic $i$ is opened, 0 otherwise, for all clinics $i \in I$.
- $x_{ij} \in \{0,1\}$: 1 if neighborhood $j$ is assigned to clinic $i$, 0 otherwise, for all clinics $i \in I$, neighborhoods $j \in J$.

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} d_j \cdot c_{ij} \cdot x_{ij}
\]

where:
- $d_j$ = demand of neighborhood $j$ (from file_0_view_0, column "Demand")
- $c_{ij}$ = distance from clinic $i$ to neighborhood $j$ (from file_1_view_0, row "Clinic" $i$, column $j$)

##### Constraints

1. **Assignment:** Each neighborhood is assigned to exactly one clinic:
   \[
   \sum_{i \in I} x_{ij} = 1, \quad \forall j \in J
   \]

2. **Clinic opening limit:** Exactly $p$ clinics are opened:
   \[
   \sum_{i \in I} y_i = p
   \]
   where $p$ is the required number of clinics to open (from file_2_view_0, row where Parameter = "NumberOfClinicsToOpen", column "Value").

3. **Assignment only to open clinics:** A neighborhood can only be assigned to an open clinic:
   \[
   x_{ij} \leq y_i, \quad \forall i \in I,\, j \in J
   \]

4. **Variable domains:**
   \[
   x_{ij} \in \{0,1\}, \quad y_i \in \{0,1\}
   \]

##### Index Sets and Data Mapping

- $I$: set of clinics, from file_1_view_0, column "Clinic"
- $J$: set of neighborhoods, from file_0_view_0, column "Neighborhood"
- $d_j$: demand for neighborhood $j$, from file_0_view_0, column "Demand"
- $c_{ij}$: distance from clinic $i$ to neighborhood $j$, from file_1_view_0, row "Clinic" $i$, column $j$
- $p$: number of clinics to open, from file_2_view_0, row where Parameter = "NumberOfClinicsToOpen", column "Value"

##### Data Mapping

- Clinics ($I$): file_1_view_0, column "Clinic"
- Neighborhoods ($J$): file_0_view_0, column "Neighborhood"
- Demand ($d_j$): file_0_view_0, column "Demand"
- Distance ($c_{ij}$): file_1_view_0, row "Clinic" $i$, column $j$
- Number of clinics to open ($p$): file_2_view_0, row Parameter = "NumberOfClinicsToOpen", column "Value"