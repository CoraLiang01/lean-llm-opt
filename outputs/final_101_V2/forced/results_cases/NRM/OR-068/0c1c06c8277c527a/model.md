#### Abstract Assignment Model

**Index Sets:**
- $M$: set of managers (from "manager_project_costs.csv", column "Unnamed: 0")
- $P$: set of projects (from "manager_project_costs.csv", columns "P1", "P2", ...)

**Parameters:**
- $c_{mp}$: cost of assigning manager $m \in M$ to project $p \in P$ (from "manager_project_costs.csv", cell at row $m$, column $p$)

**Decision Variables:**
- $x_{mp} \in \{0,1\}$: 
  - $x_{mp} = 1$ if manager $m$ is assigned to project $p$
  - $x_{mp} = 0$ otherwise

**Objective:**
\[
\min \sum_{m \in M} \sum_{p \in P} c_{mp} x_{mp}
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

- Table: "manager_project_costs.csv"
    - Index set $M$: column "Unnamed: 0" (manager identifiers)
    - Index set $P$: columns "P1", "P2", ..., "Pn" (project identifiers)
    - Parameter $c_{mp}$: cell value at row $m$ (from "Unnamed: 0"), column $p$ (from project columns)