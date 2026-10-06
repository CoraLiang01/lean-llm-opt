##### Index Sets

- $W$: Set of eligible workers (managers) not on leave, from `file_0_view_0`, where `on_leave = 0`.
- $P$: Set of projects, from `file_1_view_0`.

##### Parameters

- $c_{w,p}$: Assignment cost (in USD cents) for worker $w$ to project $p$, from `file_2_view_0`, column $p$, row $w$. If the cell is blank, assignment is forbidden.
- $\text{skill}_w$: Skill level of worker $w$, from `file_0_view_0`, column `skill`.
- $\text{required\_skill}_p$: Required skill for project $p$, from `file_1_view_0`, column `required_skill$.

##### Skill Hierarchy

Let $\text{level}(\cdot)$ map skill strings to integers: $\text{Junior}=1$, $\text{Intermediate}=2$, $\text{Senior}=3$, $\text{Expert}=4$.

##### Decision Variables

- $x_{w,p} \in \{0,1\}$: $1$ if worker $w$ is assigned to project $p$, $0$ otherwise.

##### Objective Function

$\min \sum_{w \in W} \sum_{p \in P} c_{w,p} \cdot x_{w,p}$

##### Constraints

1. **Each project assigned to exactly one worker:**

   $\sum_{w \in W} x_{w,p} = 1 \quad \forall p \in P$

2. **Each worker assigned to at most one project:**

   $\sum_{p \in P} x_{w,p} \leq 1 \quad \forall w \in W$

3. **Eligibility: Only eligible workers can be assigned:**

   $x_{w,p} = 0$ if $\text{level}(\text{skill}_w) < \text{level}(\text{required\_skill}_p)$

4. **Assignment forbidden if cost cell is blank:**

   $x_{w,p} = 0$ if $c_{w,p}$ is blank in `file_2_view_0`

5. **Workers on leave are excluded:**

   $x_{w,p} = 0$ for all $p \in P$ if $w$ has `on_leave = 1$ in `file_0_view_0`

6. **Variable domain:**

   $x_{w,p} \in \{0,1\}$

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
  "cost_matrix": {
    "table_id": "file_2_view_0",
    "row_id": "worker_id",
    "columns": ["P00", "P01", "P02", "P03", "P04", "P05"]
  }
}
```

##### Notes

- The model minimizes the total assignment cost in USD cents.
- Only workers not on leave and with sufficient skill (including hierarchy) are eligible.
- Assignments are only allowed where a cost is specified (non-blank).
- Each project is assigned to exactly one eligible worker; each worker is assigned to at most one project.