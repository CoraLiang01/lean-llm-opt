#### Abstract Assignment Model

**Index Sets:**
- $M$: set of managers (from "manager_project_costs.csv", column "Unnamed: 0")
- $P$: set of projects (from "manager_project_costs.csv", columns "P1", "P2", ...)

**Parameters:**
- $c_{mp}$: cost for manager $m \in M$ to complete project $p \in P$ (from "manager_project_costs.csv", entry at row $m$, column $p$)

**Decision Variables:**
- $x_{mp} \in \{0,1\}$: $1$ if manager $m$ is assigned to project $p$, $0$ otherwise

**Objective:**
\[
\min \sum_{m \in M} \sum_{p \in P} c_{mp} \cdot x_{mp}
\]

**Constraints:**
1. **Each manager is assigned to exactly one project:**
   \[
   \sum_{p \in P} x_{mp} = 1 \quad \forall m \in M
   \]
2. **Each project is assigned to exactly one manager:**
   \[
   \sum_{m \in M} x_{mp} = 1 \quad \forall p \in P
   \]
3. **Binary assignment variables:**
   \[
   x_{mp} \in \{0,1\} \quad \forall m \in M,\, p \in P
   \]

---

**Data Mapping:**

- Table: "manager_project_costs.csv", table_id: file_0_view_0
    - Manager identifiers: column "Unnamed: 0"
    - Project identifiers: columns "P1", "P2", "P3", ...
    - Cost parameter $c_{mp}$: value at row with manager $m$, column $p$