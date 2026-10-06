##### Objective Function:

$\quad \min \sum_{i \in \mathcal{M}} \sum_{j \in \mathcal{T}} c_{ij} \, x_{ij}$

##### Constraints

###### 1. Assignment Constraints:

$\sum_{j \in \mathcal{T}} x_{ij} = 1 \quad \forall i \in \mathcal{M}$

$\sum_{i \in \mathcal{M}} x_{ij} = 1 \quad \forall j \in \mathcal{T}$

###### 2. Variable Constraints:

$x_{ij} \in \{0,1\} \quad \forall i \in \mathcal{M}, \forall j \in \mathcal{T}$

---

##### Index Sets

- $\mathcal{M}$: Set of machines, as given by the column "assignee_id" in table_id "file_0_view_0" of "cost_12x12.csv".
- $\mathcal{T}$: Set of tasks, as given by the columns:
  - "assignment_cost_to_project_A"
  - "assignment_cost_to_project_B"
  - "assignment_cost_to_project_C"
  - "assignment_cost_to_project_D"
  - "assignment_cost_to_project_E"
  - "assignment_cost_to_project_F"
  - "assignment_cost_to_project_G"
  - "assignment_cost_to_project_H"
  - "assignment_cost_to_project_I"
  - "assignment_cost_to_project_J"
  - "assignment_cost_to_project_K"
  - "assignment_cost_to_project_L"
  in table_id "file_0_view_0" of "cost_12x12.csv".

##### Parameters

- $c_{ij}$: Machining cost of assigning machine $i$ to task $j$, as given by the value in row with "assignee_id" $i$ and column "assignment_cost_to_project_X" $j$ in table_id "file_0_view_0" of "cost_12x12.csv$.

##### Variables

- $x_{ij}$: Binary variable, equals 1 if machine $i$ is assigned to task $j$, 0 otherwise.

---

##### Data Mapping

```json
{
  "table_id": "file_0_view_0",
  "machine_index": "assignee_id",
  "task_indices": [
    "assignment_cost_to_project_A",
    "assignment_cost_to_project_B",
    "assignment_cost_to_project_C",
    "assignment_cost_to_project_D",
    "assignment_cost_to_project_E",
    "assignment_cost_to_project_F",
    "assignment_cost_to_project_G",
    "assignment_cost_to_project_H",
    "assignment_cost_to_project_I",
    "assignment_cost_to_project_J",
    "assignment_cost_to_project_K",
    "assignment_cost_to_project_L"
  ],
  "cost_parameter": "c_{ij} = \u201cassignment_cost_to_project_X\u201d value for machine i in row with assignee_id i and column assignment_cost_to_project_X"
}
```