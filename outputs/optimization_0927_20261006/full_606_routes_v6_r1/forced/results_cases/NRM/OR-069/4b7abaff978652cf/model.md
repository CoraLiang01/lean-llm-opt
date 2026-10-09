#### Abstract Assignment Model

**Index Sets:**
- $M$: set of managers (indexed by $i$)
- $P$: set of projects (indexed by $j$)

**Parameters:**
- $c_{ij}$: cost of assigning manager $i \in M$ to project $j \in P$

**Decision Variables:**
- $x_{ij} \in \{0,1\}$: 
  - $x_{ij} = 1$ if manager $i$ is assigned to project $j$, 
  - $x_{ij} = 0$ otherwise

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
   x_{ij} \in \{0,1\} \quad \forall i \in M, \forall j \in P
   \]

---

#### Data Mapping

- **Source Table:** `manager_project_costs.csv`
- **Manager Index Set ($M$):** Column `Manager`
- **Project Index Set ($P$):** Columns `Project 1 Cost`, `Project 2 Cost`, ..., `Project 11 Cost`
- **Cost Parameter ($c_{ij}$):** Entry in row for manager $i$ and column for project $j$ (column names as above)

All rows and columns from `manager_project_costs.csv` are included, with no additional filters.