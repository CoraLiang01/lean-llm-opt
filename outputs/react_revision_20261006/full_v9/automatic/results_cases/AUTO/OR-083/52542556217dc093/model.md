## Mathematical Model

**Sets**
- $W$: set of workers, $W = \{1,2,\ldots,12\}$
- $T$: set of tasks, $T = \{\text{A}, \text{B}, \text{C}, \text{D}, \text{E}, \text{F}, \text{G}, \text{H}, \text{I}, \text{J}\}$

**Parameters**
- $c_{wt}$: time required for worker $w \in W$ to complete task $t \in T$  
  (from Data Mapping: table_id = file_0_view_0, columns: "Task Time Required" (worker index), $t$)

**Decision Variables**
- $x_{wt} \in \{0,1\}$: 1 if worker $w$ is assigned to task $t$, 0 otherwise

**Objective**
\[
\min \sum_{w \in W} \sum_{t \in T} c_{wt} x_{wt}
\]

**Constraints**
1. **Each task is assigned to exactly one worker:**
   \[
   \sum_{w \in W} x_{wt} = 1 \quad \forall t \in T
   \]
2. **Each worker is assigned to at most one task:**
   \[
   \sum_{t \in T} x_{wt} \leq 1 \quad \forall w \in W
   \]
3. **Exactly 10 workers are assigned (i.e., 2 workers are not assigned):**
   \[
   \sum_{w \in W} \sum_{t \in T} x_{wt} = 10
   \]
   (This is implied by the first constraint, but included for clarity.)

4. **Variable domains:**
   \[
   x_{wt} \in \{0,1\} \quad \forall w \in W,\, t \in T
   \]

---

### Data Mapping

- $W = \{1,2,\ldots,12\}$: "Task Time Required" row indices 1–12 in table_id = file_0_view_0
- $T = \{\text{A}, \text{B}, \text{C}, \text{D}, \text{E}, \text{F}, \text{G}, \text{H}, \text{I}, \text{J}\}$: columns "A"–"J" in table_id = file_0_view_0
- $c_{wt}$: entry in row $w$ (worker $w$), column $t$ (task $t$) of table_id = file_0_view_0

---

**Summary:**  
Assign 10 out of 12 workers to 10 tasks (one per task, at most one per worker) to minimize total working hours, using the time matrix from 15.csv.