##### Objective Function:

$\quad \min \sum_{i \in W} \sum_{j \in T} t_{ij} x_{ij}$

##### Constraints

###### 1. Task Assignment Constraints:

$\sum_{i \in W} x_{ij} = 1 \quad \forall j \in T$

###### 2. Worker Assignment Constraints:

$\sum_{j \in T} x_{ij} \leq 1 \quad \forall i \in W$

###### 3. Worker Selection Constraint:

$\sum_{i \in W} \sum_{j \in T} x_{ij} = 10$

###### 4. Variable Domains:

$x_{ij} \in \{0,1\} \quad \forall i \in W, \forall j \in T$

##### Data Mapping

{
  "worker_set": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"],  // "Task Time Required" column, rows 1-12
  "task_set": ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"],  // columns A-J
  "time_matrix": {
    "table_id": "file_0_view_0",
    "row_id": "Task Time Required",
    "col_ids": ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
  }
}

Where:
- $W$ is the set of workers (IDs: "1" to "12", corresponding to "Task Time Required" values in rows 1-12 of file_0_view_0)
- $T$ is the set of tasks (IDs: "A" to "J", corresponding to columns A-J of file_0_view_0)
- $t_{ij}$ is the time required for worker $i$ to complete task $j$, from the entry in row with "Task Time Required" = $i$ and column $j$ in table_id "file_0_view_0"
- $x_{ij}$ is a binary variable indicating if worker $i$ is assigned to task $j$ (1 if assigned, 0 otherwise)

All assignments and parameters are mapped directly to the provided CSV data.