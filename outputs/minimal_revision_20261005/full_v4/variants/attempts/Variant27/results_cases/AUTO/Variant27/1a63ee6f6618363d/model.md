##### Index Sets

- $W$: Set of eligible workers (worker_id from `file_0_view_0` with on_leave = 0)
- $P$: Set of projects (project_id from `file_1_view_0`)

##### Parameters

- $c_{wp}$: Assignment cost (in USD cents) for worker $w \in W$ to project $p \in P$, from the merged cost matrices (`file_2_view_0`, `file_3_view_0`, `file_4_view_0`), matched by worker_id and project_id. If the cell is blank, assignment is forbidden.
- $s_w$: Skill level of worker $w$ (from `file_0_view_0`, column "skill")
- $r_p$: Required skill level for project $p$ (from `file_1_view_0`, column "required_skill")

##### Decision Variables

- $x_{wp} \in \{0,1\}$: $1$ if worker $w$ is assigned to project $p$, $0$ otherwise

##### Objective Function

$\min \sum_{w \in W} \sum_{p \in P} c_{wp} x_{wp}$

##### Constraints

1. **Each project assigned to exactly one worker:**

$\sum_{w \in W} x_{wp} = 1 \quad \forall p \in P$

2. **Each worker assigned to at most one project:**

$\sum_{p \in P} x_{wp} \leq 1 \quad \forall w \in W$

3. **Assignment only if worker is not on leave:**

$w \in W$ only if on\_leave = 0 (from `file_0_view_0`)

4. **Assignment only if worker skill meets or exceeds required skill:**

$x_{wp} = 0$ unless $s_w \geq r_p$ according to the order: Junior < Intermediate < Senior < Expert

5. **Assignment only if cost cell is non-blank:**

$x_{wp} = 0$ if $c_{wp}$ is blank in the merged cost matrices

6. **Variable domain:**

$x_{wp} \in \{0,1\}$

##### Data Mapping

```json
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
  ]
}
```

##### Notes

- The skill order is: Junior < Intermediate < Senior < Expert. For each assignment, $x_{wp}$ can only be $1$ if the worker's skill is at least as high as the project's required skill.
- Only workers with on\_leave = 0 are eligible.
- Only assignments with a non-blank cost cell are allowed.
- The minimum total cost is reported in USD cents.