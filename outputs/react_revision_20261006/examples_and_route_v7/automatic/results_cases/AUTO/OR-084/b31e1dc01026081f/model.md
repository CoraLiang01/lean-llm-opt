##### Objective Function:

$\quad \min \; C_{\max}$

##### Constraints

Let $T = \{1,2,\ldots,40\}$ be the set of tasks, and $P = \{1,2,3\}$ the set of CPUs. Let $b_t$ be the number of billions of instructions (BI) required by task $t \in T$, as given in the data mapping. Let $f_p$ be the frequency (in GHz) of CPU $p \in P$, with $f_1 = 1.33$, $f_2 = 2$, $f_3 = 2.66$.

Let $x_{tp} \in \{0,1\}$ be a binary variable equal to 1 if task $t$ is assigned to CPU $p$, 0 otherwise.

Let $C_{\max}$ be the completion time of the last finishing CPU.

$\textbf{Minimize:}$
$$
C_{\max}
$$

$\textbf{Subject to:}$

1. $\text{Each task is assigned to exactly one CPU:}$

$$
\sum_{p \in P} x_{tp} = 1 \quad \forall t \in T
$$

2. $\text{Each CPU can process only one task at a time (sequentially), and the total load on each CPU is the sum of its assigned tasks' processing times:}$

Let $L_p$ be the total load (in seconds) on CPU $p$:
$$
L_p = \sum_{t \in T} \frac{b_t}{f_p} x_{tp} \quad \forall p \in P
$$

3. $\text{The makespan is at least the load on each CPU:}$

$$
L_p \leq C_{\max} \quad \forall p \in P
$$

4. $\text{Variable domains:}$

$$
x_{tp} \in \{0,1\} \quad \forall t \in T, \; p \in P
$$

##### Retrieved Information

{
  "tasks": {
    "set": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31", "32", "33", "34", "35", "36", "37", "38", "39", "40"],
    "instruction_counts": {
      "file_id": "file_0_view_0",
      "column_names": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31", "32", "33", "34", "35", "36", "37", "38", "39", "40"],
      "row_label": "BI"
    }
  },
  "cpus": {
    "set": ["1", "2", "3"],
    "frequencies_GHz": {
      "1": 1.33,
      "2": 2,
      "3": 2.66
    }
  }
}