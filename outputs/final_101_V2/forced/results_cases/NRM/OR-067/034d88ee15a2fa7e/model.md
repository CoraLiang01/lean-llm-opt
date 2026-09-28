#### Abstract Mathematical Model

**Index Sets:**
- $M$: set of managers (from column "Unnamed: 0" in table_id file_0_view_0)
- $P$: set of projects (from columns "P1", "P2", "P3" in table_id file_0_view_0)

**Parameters:**
- $c_{mp}$: cost for manager $m \in M$ to complete project $p \in P$ (from table_id file_0_view_0, columns "P1", "P2", "P3")

**Decision Variables:**
- $x_{mp} \in \{0,1\}$: 
  - $x_{mp} = 1$ if manager $m$ is assigned to project $p$; $0$ otherwise

**Objective:**
\[
\min \sum_{m \in M} \sum_{p \in P} c_{mp} \, x_{mp}
\]

**Constraints:**
1. **Each manager assigned to exactly one project:**
   \[
   \sum_{p \in P} x_{mp} = 1 \quad \forall m \in M
   \]
2. **Each project assigned to exactly one manager:**
   \[
   \sum_{m \in M} x_{mp} = 1 \quad \forall p \in P
   \]
3. **Variable domain:**
   \[
   x_{mp} \in \{0,1\} \quad \forall m \in M,\, p \in P
   \]

---

**Data Mapping:**

- Table: file_0_view_0 (from "manager_project_costs.csv")
    - Manager identifiers: column "Unnamed: 0" $\rightarrow$ $M$
    - Project identifiers: columns "P1", "P2", "P3" $\rightarrow$ $P$
    - Cost parameter: $c_{mp}$ from intersection of manager row and project column

No literal values or record counts are included; all identifiers and mappings are preserved as in the source.