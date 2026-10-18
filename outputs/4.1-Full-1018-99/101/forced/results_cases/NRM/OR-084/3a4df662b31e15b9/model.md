#### Index Sets

- $T$: set of tasks (from 1 to 40), indexed by $t$.
- $P$: set of processors $\{1,2,3\}$, indexed by $p$.

#### Parameters

- $b_t$: basic instructions (in billions) required by task $t$.  
  (Data: 18.csv, column "$t$", $t=1,\ldots,40$)
- $f_p$: frequency (in GHz) of processor $p$.  
  Given: $f_1 = 1.33$, $f_2 = 2$, $f_3 = 2.66$.

#### Decision Variables

- $x_{t,p} \in \{0,1\}$: $=1$ if task $t$ is assigned to processor $p$, $0$ otherwise.
- $C_{\max} \geq 0$: completion time of the last task (makespan).

#### Objective

$$
\min \; C_{\max}
$$

#### Constraints

1. **Assignment:**  
 Each task is assigned to exactly one processor:
$$
\sum_{p \in P} x_{t,p} = 1 \quad \forall t \in T
$$

2. **Makespan (Completion Time):**  
 For each processor, the total processing time of its assigned tasks cannot exceed $C_{\max}$:
$$
\sum_{t \in T} \frac{b_t}{f_p} \cdot x_{t,p} \leq C_{\max} \quad \forall p \in P
$$

3. **Variable Domains:**  
$$
x_{t,p} \in \{0,1\} \quad \forall t \in T, \; p \in P
$$
$$
C_{\max} \geq 0
$$

---

#### Data Mapping

- Table: 18.csv (table_id: file_0_view_0)
    - Task basic instructions: columns "1" through "40" (column name = task index, value = $b_t$ in billions of instructions)
- Processor frequencies: given in the user query, not in the data file.

---

This model assigns each task to exactly one processor, and minimizes the time when the last processor finishes its assigned tasks, given the instruction counts and processor speeds.