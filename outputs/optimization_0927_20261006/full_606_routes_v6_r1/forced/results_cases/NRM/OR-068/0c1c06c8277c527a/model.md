#### Abstract Mathematical Model

**Index Sets:**
- $M$: set of managers (indexed by $i$)
- $P$: set of projects (indexed by $j$)

**Parameters:**
- $c_{ij}$: cost of assigning manager $i \in M$ to project $j \in P$

**Decision Variables:**
- $x_{ij} \in \{0,1\}$: $x_{ij} = 1$ if manager $i$ is assigned to project $j$, $0$ otherwise

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

- Table: `file_0_view_0` (from `"manager_project_costs.csv"`)
    - Manager index set $M$: values in column `"Unnamed: 0"`
    - Project index set $P$: column headers `"P1"`, `"P2"`, ..., `"P6"`
    - Cost parameter $c_{ij}$: value at row with manager $i$ and column with project $j$ (i.e., cell at intersection of `"Unnamed: 0" = i"` and column $j$)

All data from `"manager_project_costs.csv"` is used as provided, with no additional filtering.