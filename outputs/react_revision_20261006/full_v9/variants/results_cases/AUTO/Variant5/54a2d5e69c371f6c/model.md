## Mathematical Model

**Sets**
- $I$: set of teams (machines), from machine_capacity.csv and assignment_costs.csv, assignment_resources.csv
  - $I = \{\text{M1}, \text{M2}, \text{M3}, \text{M4}\}$
- $J$: set of jobs, from assignment_costs.csv and assignment_resources.csv
  - $J = \{\text{J1}, \text{J2}, \text{J3}, \text{J4}, \text{J5}, \text{J6}, \text{J7}, \text{J8}\}$

**Parameters**
- $c_{ij}$: cost of assigning job $j$ to team $i$ (from assignment_costs.csv, table_id: file_1_view_0, columns: Machine, J1–J8)
- $a_{ij}$: capacity consumed on team $i$ if job $j$ is assigned to it (from assignment_resources.csv, table_id: file_2_view_0, columns: Machine, J1–J8)
- $b_i$: available capacity of team $i$ (from machine_capacity.csv, table_id: file_0_view_0, columns: Machine, Capacity)

**Decision Variables**
- $x_{ij} \in \{0,1\}$: $=1$ if job $j$ is assigned to team $i$, $0$ otherwise

**Objective**
\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

**Constraints**
1. **Each job assigned to exactly one team:**
   \[
   \sum_{i \in I} x_{ij} = 1 \quad \forall j \in J
   \]
2. **Team capacity not exceeded:**
   \[
   \sum_{j \in J} a_{ij} x_{ij} \leq b_i \quad \forall i \in I
   \]
3. **Binary assignment:**
   \[
   x_{ij} \in \{0,1\} \quad \forall i \in I,\, j \in J
   \]

---

### Data Mapping

- **Teams ($I$):** All values in column "Machine" of machine_capacity.csv (table_id: file_0_view_0), assignment_costs.csv (file_1_view_0), assignment_resources.csv (file_2_view_0)
- **Jobs ($J$):** All columns J1–J8 in assignment_costs.csv (file_1_view_0), assignment_resources.csv (file_2_view_0)
- **$c_{ij}$:** Entry in assignment_costs.csv (file_1_view_0), row "Machine" $i$, column $j$
- **$a_{ij}$:** Entry in assignment_resources.csv (file_2_view_0), row "Machine" $i$, column $j$
- **$b_i$:** Entry in machine_capacity.csv (file_0_view_0), row "Machine" $i$, column "Capacity"

---

**All sets, parameters, and variables are defined exactly as in the source data.**