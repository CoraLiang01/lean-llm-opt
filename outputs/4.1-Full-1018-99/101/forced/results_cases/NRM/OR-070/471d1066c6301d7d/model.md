#### Abstract Mathematical Model

**Index Sets:**
- $M$: Set of managers (from column "Manager" in table_id: file_0_view_0)
- $P$: Set of projects (from columns "Project 1 Cost", "Project 2 Cost", ..., in table_id: file_0_view_0)

**Parameters:**
- $c_{mp}$: Cost for manager $m \in M$ to complete project $p \in P$ (from table_id: file_0_view_0, columns "Project * Cost")

**Decision Variables:**
- $x_{mp} \in \{0,1\}$: 
    - $x_{mp} = 1$ if manager $m$ is assigned to project $p$,
    - $x_{mp} = 0$ otherwise.

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

- **Managers ($M$):** Identifiers from column "Manager" in table_id: file_0_view_0 of "manager_project_costs.csv".
- **Projects ($P$):** Identifiers from columns "Project 1 Cost", "Project 2 Cost", ..., in table_id: file_0_view_0 of "manager_project_costs.csv".
- **Costs ($c_{mp}$):** Values from the intersection of each manager row and project cost column in table_id: file_0_view_0 of "manager_project_costs.csv".

No additional filters or eligibility conditions were applied; all records and columns from the source were used as returned by CSVQA.