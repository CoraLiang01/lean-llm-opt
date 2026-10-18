#### Abstract Assignment Model

**Index Sets:**
- $M$: set of machines (from column "Machine" in file_0_view_0)
- $T$: set of tasks (from columns "A", "B", ..., "L" in file_0_view_0)

**Parameters:**
- $c_{i,j}$: cost of assigning machine $i \in M$ to task $j \in T$ (from file_0_view_0, entry at row with "Machine" = $i$, column $j$)

**Decision Variables:**
- $x_{i,j} \in \{0,1\}$: $1$ if machine $i$ is assigned to task $j$, $0$ otherwise, for all $i \in M$, $j \in T$

**Objective:**
\[
\min \sum_{i \in M} \sum_{j \in T} c_{i,j} \cdot x_{i,j}
\]

**Constraints:**
1. **Each machine is assigned to exactly one task:**
   \[
   \sum_{j \in T} x_{i,j} = 1 \quad \forall i \in M
   \]
2. **Each task is assigned to exactly one machine:**
   \[
   \sum_{i \in M} x_{i,j} = 1 \quad \forall j \in T
   \]
3. **Binary assignment:**
   \[
   x_{i,j} \in \{0,1\} \quad \forall i \in M,\, j \in T
   \]

---

**Data Mapping:**

- Table: cost_12x12.csv (table_id: file_0_view_0)
    - Machine index set $M$: column "Machine"
    - Task index set $T$: columns "A", "B", ..., "L"
    - Cost parameter $c_{i,j}$: value at row with "Machine" = $i$, column $j$