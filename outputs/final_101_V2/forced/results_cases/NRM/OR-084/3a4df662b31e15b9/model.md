#### Abstract Mathematical Model

**Index Sets:**
- $T$: set of tasks (from 1 to 40)
- $P$: set of processors, $P = \{1,2,3\}$

**Parameters:**
- $b_t$: basic instructions (in billions) required by task $t \in T$ (from column "t" in table_id file_0_view_0, where $t$ is the task number as string)
- $f_p$: frequency (in GHz) of processor $p \in P$
    - $f_1 = 1.33$
    - $f_2 = 2$
    - $f_3 = 2.66$

**Decision Variables:**
- $x_{t,p} \in \{0,1\}$: $=1$ if task $t$ is assigned to processor $p$, $0$ otherwise
- $C_{\max} \geq 0$: makespan (completion time of the last task)

**Objective:**
\[
\min \; C_{\max}
\]

**Constraints:**

1. **Assignment:** Each task is assigned to exactly one processor:
   \[
   \sum_{p \in P} x_{t,p} = 1 \quad \forall t \in T
   \]

2. **Makespan:** The makespan is at least the total processing time on each processor:
   \[
   \sum_{t \in T} \frac{b_t}{f_p} x_{t,p} \leq C_{\max} \quad \forall p \in P
   \]

3. **Variable Domains:**
   \[
   x_{t,p} \in \{0,1\} \quad \forall t \in T, \; p \in P
   \]
   \[
   C_{\max} \geq 0
   \]

---

#### Data Mapping

- Table: 18.csv, table_id: file_0_view_0
    - Task set $T$ and parameter $b_t$ are given by columns "1", "2", ..., "40" in the row where "Process" = "BI".
- Processor frequencies $f_p$ are given in the user query.

No other data sources are used.