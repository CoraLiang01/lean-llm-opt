##### Objective Function:

$\quad \min \; C_{\max}$

##### Constraints

###### 1. Each task is assigned to exactly one CPU:

$\sum_{k \in K} x_{ik} = 1 \quad \forall i \in I$

###### 2. CPU load definition (completion time for each CPU):

$\sum_{i \in I} \frac{b_i}{f_k} x_{ik} \leq C_{\max} \quad \forall k \in K$

###### 3. Variable domains:

$x_{ik} \in \{0,1\} \quad \forall i \in I, \; k \in K$

$C_{\max} \geq 0$

##### Index Sets and Parameters

- $I$: Set of tasks, $I = \{1, 2, \ldots, 40\}$
- $K$: Set of CPUs, $K = \{1, 2, 3\}$
- $b_i$: Basic instructions (in billions) required by task $i$, from 18.csv, column $i$, row where Process = 'BI', table_id = file_0_view_0
- $f_k$: Frequency (in GHz) of CPU $k$, with $f_1 = 1.33$, $f_2 = 2$, $f_3 = 2.66$

##### Decision Variables

- $x_{ik} = \begin{cases} 1 & \text{if task } i \text{ is assigned to CPU } k \\ 0 & \text{otherwise} \end{cases}$
- $C_{\max}$: The maximum completion time across all CPUs

##### Retrieved Information

{
  "tasks": {
    "source": "file_0_view_0",
    "columns": ["1", "2", ..., "40"],
    "parameter": "b_i",
    "row": {"Process": "BI"}
  },
  "CPUs": {
    "K": [1, 2, 3],
    "frequencies": {
      "1": 1.33,
      "2": 2,
      "3": 2.66
    }
  }
}