##### Objective Function:

$\quad \min \sum_{i \in \mathcal{M}} \sum_{j \in \mathcal{T}} c_{ij} \, x_{ij}$

##### Constraints

###### 1. Assignment Constraints:

$\sum_{j \in \mathcal{T}} x_{ij} = 1 \quad \forall i \in \mathcal{M}$

$\sum_{i \in \mathcal{M}} x_{ij} = 1 \quad \forall j \in \mathcal{T}$

###### 2. Variable Constraints:

$x_{ij} \in \{0,1\} \quad \forall i \in \mathcal{M}, \; j \in \mathcal{T}$

##### Index Sets

- $\mathcal{M}$: Set of machines, as given by the values in column "assignee_id" of table_id "file_0_view_0" in "cost_12x12.csv".
- $\mathcal{T}$: Set of tasks, as given by the columns "assignment_cost_to_project_A", "assignment_cost_to_project_B", ..., "assignment_cost_to_project_L" of table_id "file_0_view_0" in "cost_12x12.csv".

##### Parameters

- $c_{ij}$: Machining cost of assigning machine $i$ to task $j$, given by the value in row with "assignee_id" $i$ and column $j$ ("assignment_cost_to_project_X") in table_id "file_0_view_0" of "cost_12x12.csv".

##### Decision Variables

- $x_{ij} = \begin{cases} 1 & \text{if machine } i \text{ is assigned to task } j \\ 0 & \text{otherwise} \end{cases}$

##### Data Mapping

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
  "cost_parameter": "c_{ij} = \text{value at row with assignee_id } i \text{ and column } j"
}