##### Objective Function:

$\quad \min \sum_{w \in W} \sum_{p \in P} c_{w,p} \, x_{w,p}$

##### Constraints

###### 1. Project Assignment

$\sum_{w \in W} x_{w,p} = 1 \quad \forall p \in P$

###### 2. Worker Assignment

$\sum_{p \in P} x_{w,p} \leq 1 \quad \forall w \in W$

###### 3. Leave Exclusion

$x_{w,p} = 0 \quad \forall w \in W_{\text{leave}}, \forall p \in P$

###### 4. Skill Feasibility

$x_{w,p} = 0 \quad \forall (w,p): \text{skill}(w) < \text{required\_skill}(p)$

###### 5. Forbidden Assignments

$x_{w,p} = 0 \quad \forall (w,p): c_{w,p} \text{ is blank}$

###### 6. Variable Domain

$x_{w,p} \in \{0,1\} \quad \forall w \in W, \forall p \in P$

##### Data Mapping

- $W$: Set of worker IDs from file_0_view_0 where on_leave = 0.
- $W_{\text{leave}}$: Set of worker IDs from file_0_view_0 where on_leave = 1.
- $P$: Set of project IDs from file_1_view_0.
- $\text{skill}(w)$: Skill level of worker $w$ from file_0_view_0, column "skill".
- $\text{required\_skill}(p)$: Required skill for project $p$ from file_1_view_0, column "required_skill".
- $c_{w,p}$: Assignment cost from file_2_view_0, row "worker_id" = $w$, column $p$; blank cells forbid assignment.
- $x_{w,p}$: Binary variable, 1 if worker $w$ is assigned to project $p$, 0 otherwise.

##### Skill Order

$\text{Junior} < \text{Intermediate} < \text{Senior} < \text{Expert}$

##### Indices

- $w \in W$: All worker IDs with on_leave = 0 from file_0_view_0.
- $p \in P$: All project IDs from file_1_view_0.

##### Objective Units

- The objective value is in USD cents.

##### Source Data Mapping

{
  "workers_table": {
    "table_id": "file_0_view_0",
    "columns": ["worker_id", "skill", "on_leave"]
  },
  "projects_table": {
    "table_id": "file_1_view_0",
    "columns": ["project_id", "required_skill"]
  },
  "cost_matrix": {
    "table_id": "file_2_view_0",
    "row_id": "worker_id",
    "columns": ["P00", "P01", "P02", "P03", "P04", "P05"]
  }
}