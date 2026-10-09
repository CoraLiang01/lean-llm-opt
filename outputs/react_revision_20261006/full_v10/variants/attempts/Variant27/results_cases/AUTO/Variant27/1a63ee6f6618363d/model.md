##### Objective Function:

$\min \sum_{w \in W} \sum_{p \in P} c_{wp} \, x_{wp}$

where $c_{wp}$ is the assignment cost in USD cents for worker $w$ to project $p$, as given in the cost matrices (blank cells: assignment forbidden).

##### Constraints

1. **Project Assignment:**  
$\sum_{w \in W} x_{wp} = 1 \quad \forall p \in P$  
(Each project is assigned exactly one worker.)

2. **Worker Assignment:**  
$\sum_{p \in P} x_{wp} \leq 1 \quad \forall w \in W$  
(Each worker is assigned to at most one project.)

3. **Eligibility:**  
$x_{wp} = 0$ if worker $w$ is on leave ($\text{on\_leave}=1$ in file_0_view_0),  
or if $c_{wp}$ is blank in the cost matrix (file_2_view_0, file_3_view_0, file_4_view_0),  
or if worker $w$'s skill (file_0_view_0) is less than the required_skill for project $p$ (file_1_view_0), using the order Junior < Intermediate < Senior < Expert.

4. **Variable Domain:**  
$x_{wp} \in \{0,1\} \quad \forall w \in W,\, p \in P$

##### Retrieved Information

{
  "workers": {
    "table_id": "file_0_view_0",
    "columns": ["worker_id", "skill", "on_leave"]
  },
  "projects": {
    "table_id": "file_1_view_0",
    "columns": ["project_id", "required_skill"]
  },
  "cost_matrix": [
    {
      "table_id": "file_2_view_0",
      "row_id": "worker_id",
      "columns": ["P00", "P01", "P02", "P03", "P04", "P05", "P06", "P07", "P08", "P09"]
    },
    {
      "table_id": "file_3_view_0",
      "row_id": "worker_id",
      "columns": ["P00", "P01", "P02", "P03", "P04", "P05", "P06", "P07", "P08", "P09"]
    },
    {
      "table_id": "file_4_view_0",
      "row_id": "worker_id",
      "columns": ["P00", "P01", "P02", "P03", "P04", "P05", "P06", "P07", "P08", "P09"]
    }
  ],
  "skill_order": ["Junior", "Intermediate", "Senior", "Expert"]
}

Where:
- $W$ is the set of worker_ids from file_0_view_0 with on_leave=0 and present in any cost matrix row.
- $P$ is the set of project_ids from file_1_view_0.
- $c_{wp}$ is the value in the cost matrix for worker $w$ and project $p$ (from the union of file_2_view_0, file_3_view_0, file_4_view_0), if present and not blank; otherwise, assignment is forbidden.
- A worker $w$ can only be assigned to project $p$ if their skill (from file_0_view_0) is greater than or equal to the required_skill for $p$ (from file_1_view_0), using the order Junior < Intermediate < Senior < Expert.