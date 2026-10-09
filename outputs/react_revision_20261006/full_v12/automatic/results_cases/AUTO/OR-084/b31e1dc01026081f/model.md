## Mathematical Model

**Sets**
- $T = \{1,2,\ldots,40\}$: set of tasks (from columns "1" to "40" in 18.csv)
- $P = \{1,2,3\}$: set of processors

**Parameters**
- $b_t$: basic instructions (in billions) required for task $t \in T$ (from 18.csv, table_id: file_0_view_0, column $t$)
- $f_p$: frequency (in GHz) of processor $p \in P$:
    - $f_1 = 1.33$
    - $f_2 = 2$
    - $f_3 = 2.66$

**Decision Variables**
- $x_{t,p} \in \{0,1\}$: 1 if task $t$ is assigned to processor $p$, 0 otherwise
- $C_{\max} \geq 0$: completion time of the last task (to be minimized)

**Objective**
$$
\min C_{\max}
$$

**Constraints**

1. **Each task assigned to exactly one processor:**
$$
\sum_{p \in P} x_{t,p} = 1 \quad \forall t \in T
$$

2. **Processor load and makespan:**
$$
\sum_{t \in T} \frac{b_t}{f_p} x_{t,p} \leq C_{\max} \quad \forall p \in P
$$

3. **Variable domains:**
$$
x_{t,p} \in \{0,1\} \quad \forall t \in T,\, p \in P
$$
$$
C_{\max} \geq 0
$$

---

### Data Mapping

- $T$: task indices from columns "1" to "40" in table_id: file_0_view_0, 18.csv
- $b_t$: value in column $t$ of table_id: file_0_view_0, row "BI"
- $P$: $\{1,2,3\}$, corresponding to frequencies $f_1=1.33$, $f_2=2$, $f_3=2.66$ (GHz)
- $x_{t,p}$: binary assignment variable for task $t$ to processor $p$
- $C_{\max}$: continuous, makespan to minimize

**All parameters and sets are mapped directly from the current 18.csv as described above.**