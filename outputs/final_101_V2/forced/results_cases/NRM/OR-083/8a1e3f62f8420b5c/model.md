#### Abstract Mathematical Model

**Index Sets:**
- $W$: set of all workers (from 15.csv, 12 workers)
- $T$: set of all tasks (from 15.csv, 10 tasks)

**Parameters:**
- $c_{wt}$: time required for worker $w \in W$ to complete task $t \in T$ (from 15.csv, column for worker $w$, row for task $t$)

**Decision Variables:**
- $x_{wt} \in \{0,1\}$: $=1$ if worker $w$ is assigned to task $t$, $0$ otherwise
- $y_w \in \{0,1\}$: $=1$ if worker $w$ is selected (assigned to any task), $0$ otherwise

**Objective:**
\[
\min \sum_{w \in W} \sum_{t \in T} c_{wt} x_{wt}
\]

**Constraints:**
1. **Each task is assigned to exactly one worker:**
   \[
   \sum_{w \in W} x_{wt} = 1 \quad \forall t \in T
   \]
2. **Each selected worker is assigned to at most one task:**
   \[
   \sum_{t \in T} x_{wt} \leq y_w \quad \forall w \in W
   \]
3. **Exactly 10 workers are selected:**
   \[
   \sum_{w \in W} y_w = 10
   \]
4. **Each worker is assigned to at most one task:**
   \[
   \sum_{t \in T} x_{wt} \leq 1 \quad \forall w \in W
   \]
5. **Variable domains:**
   \[
   x_{wt} \in \{0,1\} \quad \forall w \in W, t \in T
   \]
   \[
   y_w \in \{0,1\} \quad \forall w \in W
   \]

---

#### Data Mapping

- **Table:** 15.csv
- **Worker Index Set $W$:** Column headers "A", "B", ..., "J", plus two more as per the 12 workers in the file (see all column names except "Task Time Required")
- **Task Index Set $T$:** Row labels in "Task Time Required" column (excluding header row)
- **Parameter $c_{wt}$:** Value at row with "Task Time Required" = $t$, column $w$ (for all $w \in W$, $t \in T$)