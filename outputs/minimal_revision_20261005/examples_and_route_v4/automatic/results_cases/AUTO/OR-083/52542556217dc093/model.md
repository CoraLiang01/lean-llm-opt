##### Objective Function:

$\quad \min \sum_{w \in W} \sum_{t \in T} c_{wt} \, x_{wt}$

##### Constraints

###### 1. Each task is assigned to exactly one worker:
$\sum_{w \in W} x_{wt} = 1 \quad \forall t \in T$

###### 2. Each selected worker is assigned to at most one task:
$\sum_{t \in T} x_{wt} \leq 1 \quad \forall w \in W$

###### 3. Exactly 10 workers are selected:
$\sum_{w \in W} y_w = 10$

###### 4. Linking constraint: a worker can only be assigned if selected:
$\sum_{t \in T} x_{wt} \leq y_w \cdot |T| \quad \forall w \in W$

###### 5. Variable domains:
$x_{wt} \in \{0,1\} \quad \forall w \in W,\, t \in T$

$y_w \in \{0,1\} \quad \forall w \in W$

##### Index Sets

- $W$: Set of workers, as listed in the columns "A", "B", ..., "L" of 15.csv (excluding header).
- $T$: Set of tasks, as listed in the rows "1", "2", ..., "10" of the "Task Time Required" column of 15.csv.

##### Parameters

- $c_{wt}$: Time required for worker $w$ to complete task $t$, from 15.csv, table_id: file_0_view_0, with $w$ in columns "A"–"L" and $t$ in rows where "Task Time Required" = "1"–"10".

##### Variables

- $x_{wt}$: 1 if worker $w$ is assigned to task $t$, 0 otherwise.
- $y_w$: 1 if worker $w$ is selected (assigned to any task), 0 otherwise.

##### Data Mapping

{
  "table_id": "file_0_view_0",
  "worker_columns": ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"],
  "task_rows": [
    {"Task Time Required": "1"},
    {"Task Time Required": "2"},
    {"Task Time Required": "3"},
    {"Task Time Required": "4"},
    {"Task Time Required": "5"},
    {"Task Time Required": "6"},
    {"Task Time Required": "7"},
    {"Task Time Required": "8"},
    {"Task Time Required": "9"},
    {"Task Time Required": "10"}
  ],
  "parameter": "c_{wt} = \text{entry at } (w, t) \text{ in 15.csv, table_id file_0_view_0}"
}