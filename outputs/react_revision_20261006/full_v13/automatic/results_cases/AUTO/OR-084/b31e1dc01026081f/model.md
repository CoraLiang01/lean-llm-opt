## Symbolic Mathematical Model

**Sets**
- $T = \{1,2,\ldots,40\}$: set of tasks (task indices as in 18.csv columns "1" to "40")
- $P = \{1,2,3\}$: set of processors

**Parameters**  
(Data Mapping: 18.csv, table_id = file_0_view_0)
- $b_t$: basic instructions (in billions, BI) required for task $t \in T$  
  (Data: $b_t$ is the value in column $t$ of row "BI" in 18.csv)
- $f_p$: frequency (in GHz) of processor $p \in P$  
  ($f_1 = 1.33$, $f_2 = 2$, $f_3 = 2.66$)

**Decision Variables**
- $x_{tp} \in \{0,1\}$: 1 if task $t$ is assigned to processor $p$, 0 otherwise
- $C_{\max} \geq 0$: completion time of the last finishing task (to be minimized)

**Objective**
$$
\min C_{\max}
$$

**Constraints**

1. **Each task assigned to exactly one processor:**
$$
\sum_{p \in P} x_{tp} = 1 \qquad \forall t \in T
$$

2. **Processor sequential execution (no parallelism per processor):**  
Let $S_p$ be the set of tasks assigned to processor $p$ (i.e., $x_{tp}=1$).  
The total processing time on processor $p$ is:
$$
\sum_{t \in T} \frac{b_t}{f_p} x_{tp} \leq C_{\max} \qquad \forall p \in P
$$

3. **Variable domains:**
$$
x_{tp} \in \{0,1\} \qquad \forall t \in T,\, p \in P
$$
$$
C_{\max} \geq 0
$$

---

## Data Mapping

- $T$: task indices from 18.csv columns "1" to "40"
- $b_t$: value in 18.csv, table_id = file_0_view_0, column $t$, row "BI"
- $P$: $\{1,2,3\}$, corresponding to processors with frequencies $f_1=1.33$, $f_2=2$, $f_3=2.66$ GHz

---

**Summary:**  
Assign each task to exactly one processor. Each processor executes its assigned tasks sequentially (order does not affect makespan since tasks are non-preemptive and independent). The objective is to minimize the maximum total processing time across all processors, where each task's processing time on processor $p$ is $b_t/f_p$.