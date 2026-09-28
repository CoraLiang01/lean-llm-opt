#### Abstract Assignment Model

**Index Sets:**
- $M$: set of machines (from column "Machine" in cost_12x12.csv)
- $T$: set of tasks (from columns "A", "B", ..., "L" in cost_12x12.csv)

**Parameters:**
- $c_{ij}$: cost of assigning machine $i \in M$ to task $j \in T$ (from table_id: file_0_view_0, columns: "Machine", "A"-"L")

**Decision Variables:**
- $x_{ij} \in \{0,1\}$: 
  - $x_{ij} = 1$ if machine $i$ is assigned to task $j$, 
  - $x_{ij} = 0$ otherwise

**Objective:**
\[
\min \sum_{i \in M} \sum_{j \in T} c_{ij} x_{ij}
\]

**Constraints:**
1. **Each machine assigned to exactly one task:**
   \[
   \sum_{j \in T} x_{ij} = 1 \quad \forall i \in M
   \]
2. **Each task assigned to exactly one machine:**
   \[
   \sum_{i \in M} x_{ij} = 1 \quad \forall j \in T
   \]
3. **Binary assignment:**
   \[
   x_{ij} \in \{0,1\} \quad \forall i \in M, \forall j \in T
   \]

---

#### Data Mapping

- **Table:** cost_12x12.csv (table_id: file_0_view_0)
- **Machine index set $M$:** column "Machine"
- **Task index set $T$:** columns "A", "B", ..., "L"
- **Cost parameter $c_{ij}$:** value in row with "Machine" = $i$, column $j$ ("A"-"L")