##### Objective Function:

$\quad \min \sum_{i \in \mathcal{M}} \sum_{j \in \mathcal{T}} c_{ij} \, x_{ij}$

##### Constraints

###### 1. Each machine is assigned to exactly one task:

$\sum_{j \in \mathcal{T}} x_{ij} = 1 \quad \forall i \in \mathcal{M}$

###### 2. Each task is assigned to exactly one machine:

$\sum_{i \in \mathcal{M}} x_{ij} = 1 \quad \forall j \in \mathcal{T}$

###### 3. Variable domains:

$x_{ij} \in \{0,1\} \quad \forall i \in \mathcal{M}, \; j \in \mathcal{T}$

##### Index Sets

- $\mathcal{M}$: Set of machines, as given by the column "assignee_id" in table_id "file_0_view_0".
- $\mathcal{T}$: Set of tasks, as given by the columns "assignment_cost_to_project_A", "assignment_cost_to_project_B", ..., "assignment_cost_to_project_L" in table_id "file_0_view_0".

##### Parameters

- $c_{ij}$: Assignment cost of machine $i$ to task $j$, given by the value in row with "assignee_id" $i$ and column "assignment_cost_to_project_X" (where $j$ corresponds to project $X$) in table_id "file_0_view_0".

##### Decision Variables

- $x_{ij} = \begin{cases} 1 & \text{if machine } i \text{ is assigned to task } j \\ 0 & \text{otherwise} \end{cases}$

##### Data Mapping

{
  "table_id": "file_0_view_0",
  "machine_id_column": "assignee_id",
  "task_cost_columns": [
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
  ]
}