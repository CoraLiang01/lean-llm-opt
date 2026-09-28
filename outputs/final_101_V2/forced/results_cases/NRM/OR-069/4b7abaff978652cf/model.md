#### Abstract Mathematical Model

**Index Sets:**
- $M$: set of managers (from column "Manager" in table_id file_0_view_0)
- $P$: set of projects (from columns "Project 1 Cost", ..., "Project 11 Cost" in table_id file_0_view_0)

**Parameters:**
- $c_{ij}$: assignment cost of manager $i \in M$ to project $j \in P$ (from table_id file_0_view_0, columns "Project * Cost")

**Decision Variables:**
- $x_{ij} \in \{0,1\}$: 
    - $x_{ij} = 1$ if manager $i$ is assigned to project $j$,
    - $x_{ij} = 0$ otherwise.

**Objective:**
\[
\min \sum_{i \in M} \sum_{j \in P} c_{ij} x_{ij}
\]

**Constraints:**
1. **Each manager assigned to exactly one project:**
   \[
   \sum_{j \in P} x_{ij} = 1 \quad \forall i \in M
   \]
2. **Each project assigned to exactly one manager:**
   \[
   \sum_{i \in M} x_{ij} = 1 \quad \forall j \in P
   \]
3. **Binary assignment:**
   \[
   x_{ij} \in \{0,1\} \quad \forall i \in M,\, j \in P
   \]

---

**Data Mapping:**

- **Manager set $M$:** All unique values in column "Manager" of table_id file_0_view_0 (from "manager_project_costs.csv").
- **Project set $P$:** All columns with names "Project * Cost" in table_id file_0_view_0 (from "manager_project_costs.csv").
- **Cost parameter $c_{ij}$:** Value in row with "Manager" $i$ and column "Project $j$ Cost" in table_id file_0_view_0 (from "manager_project_costs.csv").