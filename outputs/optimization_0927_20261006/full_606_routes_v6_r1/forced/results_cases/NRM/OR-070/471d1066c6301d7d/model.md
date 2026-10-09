#### Abstract Assignment Model

**Index Sets:**
- $M$: set of managers (from column "Manager" in file_0_view_0)
- $P$: set of projects (from columns "Project 1 Cost", "Project 2 Cost", ..., "Project 7 Cost" in file_0_view_0)

**Parameters:**
- $c_{mp}$: cost for manager $m \in M$ to complete project $p \in P$ (from the corresponding cell in file_0_view_0, with $m$ identified by "Manager" and $p$ by project column)

**Decision Variables:**
- $x_{mp} \in \{0,1\}$: 
    - $x_{mp} = 1$ if manager $m$ is assigned to project $p$
    - $x_{mp} = 0$ otherwise

**Objective:**
\[
\min \sum_{m \in M} \sum_{p \in P} c_{mp} \cdot x_{mp}
\]

**Constraints:**

1. **Each project is assigned to exactly one manager:**
   \[
   \sum_{m \in M} x_{mp} = 1 \quad \forall p \in P
   \]

2. **Each manager is assigned to exactly one project:**
   \[
   \sum_{p \in P} x_{mp} = 1 \quad \forall m \in M
   \]

3. **Binary assignment variables:**
   \[
   x_{mp} \in \{0,1\} \quad \forall m \in M,\, p \in P
   \]

---

**Data Mapping:**

- Table: file_0_view_0 (from manager_project_costs.csv)
    - Index set $M$: column "Manager"
    - Index set $P$: columns "Project 1 Cost", "Project 2 Cost", ..., "Project 7 Cost"
    - Parameter $c_{mp}$: value in row with "Manager" = $m$ and column = $p$ (project cost column)
- All rows and columns from file_0_view_0 are included (FALLBACK_FULL_DATA). No additional filters applied.