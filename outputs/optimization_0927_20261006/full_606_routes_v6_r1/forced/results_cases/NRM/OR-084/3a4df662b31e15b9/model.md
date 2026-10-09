#### Index Sets

- $T$: set of tasks (from 1 to 40), indexed by $t$
- $P$: set of processors (1, 2, 3), indexed by $p$

#### Parameters

- $b_t$: number of basic instructions (in billions) required by task $t$  
  (from 18.csv, column "$t$", row where "Process" = "BI")
- $f_p$: frequency (in GHz) of processor $p$  
  (given: $f_1 = 1.33$, $f_2 = 2$, $f_3 = 2.66$)

#### Decision Variables

- $x_{t,p} \in \{0,1\}$: 1 if task $t$ is assigned to processor $p$, 0 otherwise
- $C_p \geq 0$: completion time of processor $p$
- $C_{\max} \geq 0$: makespan (completion time of the last finishing processor)

#### Objective

$$
\min \; C_{\max}
$$

#### Constraints

1. **Task Assignment:**  
   Each task is assigned to exactly one processor:
   $$
   \sum_{p \in P} x_{t,p} = 1 \quad \forall t \in T
   $$

2. **Processor Completion Time:**  
   The total time spent on each processor is the sum of its assigned tasks' processing times:
   $$
   C_p = \sum_{t \in T} \frac{b_t}{f_p} \cdot x_{t,p} \quad \forall p \in P
   $$

3. **Makespan Definition:**  
   The makespan is at least as large as each processor's completion time:
   $$
   C_{\max} \geq C_p \quad \forall p \in P
   $$

4. **Variable Domains:**  
   $$
   x_{t,p} \in \{0,1\} \quad \forall t \in T, \; p \in P
   $$
   $$
   C_p \geq 0 \quad \forall p \in P
   $$
   $$
   C_{\max} \geq 0
   $$

---

#### Data Mapping

- Table: 18.csv (table_id: file_0_view_0)
    - Task set $T$ and instruction counts $b_t$ from columns "1" through "40", row where "Process" = "BI"
- Processor set $P$ and frequencies $f_p$ are given in the user query (not in the CSV)