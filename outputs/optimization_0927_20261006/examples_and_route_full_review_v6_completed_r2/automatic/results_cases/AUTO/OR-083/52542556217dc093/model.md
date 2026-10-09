##### Sets and Indices

- Let $W = \{A, B, C, D, E, F, G, H, I, J, K, L\}$ be the set of 12 workers (from the CSV, columns A–J; assume K and L are the remaining two workers, as only 10 columns are present in the data. If only 10 workers are available, adjust accordingly).
- Let $T = \{1,2,3,4,5,6,7,8,9,10\}$ be the set of 10 tasks (from the CSV, "Task Time Required" 1–10).

##### Parameters

Let $c_{w,t}$ be the time required for worker $w$ to complete task $t$, as given in the CSV:

\[
\begin{array}{c|cccccccccc}
\text{Task} & A & B & C & D & E & F & G & H & I & J \\
\hline
1 & 9 & 4 & 3 & 7 & 6 & 5 & 6 & 3 & 7 & 5 \\
2 & 4 & 6 & 5 & 6 & 4 & 5 & 3 & 8 & 7 & 6 \\
3 & 5 & 4 & 7 & 5 & 6 & 6 & 5 & 8 & 6 & 9 \\
4 & 7 & 5 & 2 & 3 & 7 & 8 & 5 & 6 & 8 & 5 \\
5 & 10 & 6 & 7 & 4 & 5 & 4 & 4 & 5 & 9 & 7 \\
6 & 6 & 7 & 6 & 3 & 9 & 5 & 7 & 4 & 3 & 4 \\
7 & 8 & 8 & 5 & 9 & 5 & 7 & 5 & 9 & 5 & 3 \\
8 & 7 & 4 & 8 & 8 & 6 & 7 & 5 & 7 & 7 & 7 \\
9 & 5 & 6 & 8 & 7 & 7 & 8 & 7 & 8 & 4 & 5 \\
10 & 8 & 7 & 9 & 5 & 8 & 5 & 9 & 9 & 3 & 4 \\
\end{array}
\]

##### Decision Variables

- $x_{w,t} \in \{0,1\}$: 1 if worker $w$ is assigned to task $t$, 0 otherwise.
- $y_w \in \{0,1\}$: 1 if worker $w$ is selected (assigned to any task), 0 otherwise.

##### Objective Function

\[
\min \sum_{w \in W} \sum_{t \in T} c_{w,t} \cdot x_{w,t}
\]

##### Constraints

1. **Each task is assigned to exactly one worker:**
   \[
   \sum_{w \in W} x_{w,t} = 1 \quad \forall t \in T
   \]

2. **Each selected worker is assigned to at most one task:**
   \[
   \sum_{t \in T} x_{w,t} \leq y_w \quad \forall w \in W
   \]

3. **Exactly 10 workers are selected:**
   \[
   \sum_{w \in W} y_w = 10
   \]

4. **Each selected worker is assigned to exactly one task (since there are 10 tasks and 10 selected workers):**
   \[
   \sum_{t \in T} x_{w,t} = y_w \quad \forall w \in W
   \]

5. **Variable domains:**
   \[
   x_{w,t} \in \{0,1\} \quad \forall w \in W, t \in T
   \]
   \[
   y_w \in \{0,1\} \quad \forall w \in W
   \]

##### Retrieved Information

{
  "workers": ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"],
  "tasks": [1,2,3,4,5,6,7,8,9,10],
  "time": {
    "A": [9,4,5,7,10,6,8,7,5,8],
    "B": [4,6,4,5,6,7,8,4,6,7],
    "C": [3,5,7,2,7,6,5,8,8,9],
    "D": [7,6,5,3,4,3,9,8,7,5],
    "E": [6,4,6,7,5,9,5,6,7,8],
    "F": [5,5,6,8,4,5,7,7,8,5],
    "G": [6,3,5,5,4,7,5,5,7,9],
    "H": [3,8,8,6,5,4,9,7,8,9],
    "I": [7,7,6,8,9,3,5,7,4,3],
    "J": [5,6,9,5,7,4,3,7,5,4]
    // If K and L are present, add their times here; otherwise, only 10 workers are available.
  }
}

**Note:** If only 10 workers (A–J) are available, then all must be selected and the model simplifies to a standard assignment problem. If 12 workers are available, use the above model to select the best 10.