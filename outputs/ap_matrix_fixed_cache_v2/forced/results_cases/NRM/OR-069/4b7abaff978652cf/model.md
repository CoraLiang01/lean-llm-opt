#### Abstract Assignment Model

**Index Sets:**
- $M$: set of managers (indexed by $i$)
- $P$: set of projects (indexed by $j$)

**Parameters:**
- $c_{ij}$: assignment cost of manager $i \in M$ to project $j \in P$

**Decision Variables:**
- $x_{ij} \in \{0,1\}$: 
  - $x_{ij} = 1$ if manager $i$ is assigned to project $j$; $0$ otherwise

**Objective:**
\[
\min \sum_{i \in M} \sum_{j \in P} c_{ij} x_{ij}
\]

**Constraints:**
1. **Each manager is assigned to exactly one project:**
   \[
   \sum_{j \in P} x_{ij} = 1 \quad \forall i \in M
   \]
2. **Each project is assigned to exactly one manager:**
   \[
   \sum_{i \in M} x_{ij} = 1 \quad \forall j \in P
   \]
3. **Binary assignment variables:**
   \[
   x_{ij} \in \{0,1\} \quad \forall i \in M,\, j \in P
   \]

---

**Data Mapping:**

- Table: `manager_project_costs.csv` (table_id: file_0_view_0)
- Manager index set $M$: column `Manager`
- Project index set $P$: columns `Project 1 Cost`, `Project 2 Cost`, ..., `Project 11 Cost`
- Assignment cost parameter $c_{ij}$: value in row $i$ (manager) and column $j$ (project cost column)