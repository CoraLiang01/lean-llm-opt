##### Objective Function

$\quad \min \sum_{i \in \{\text{MA}, \text{MB}, \text{MC}\}} \sum_{j \in \{\text{P1}, \text{P2}, \text{P3}\}} c_{ij} x_{ij}$

where $c_{ij}$ is the cost for manager $i$ to manage project $j$, and $x_{ij}$ is a binary variable equal to 1 if manager $i$ is assigned to project $j$, 0 otherwise.

##### Constraints

###### 1. Each manager is assigned to exactly one project:

$\sum_{j \in \{\text{P1}, \text{P2}, \text{P3}\}} x_{ij} = 1 \quad \forall i \in \{\text{MA}, \text{MB}, \text{MC}\}$

###### 2. Each project is assigned to exactly one manager:

$\sum_{i \in \{\text{MA}, \text{MB}, \text{MC}\}} x_{ij} = 1 \quad \forall j \in \{\text{P1}, \text{P2}, \text{P3}\}$

###### 3. Variable Domains:

$x_{ij} \in \{0,1\} \quad \forall i \in \{\text{MA}, \text{MB}, \text{MC}\},\ j \in \{\text{P1}, \text{P2}, \text{P3}\}$

##### Retrieved Information

{
  "managers": [
    {
      "id": "MA",
      "manager_professional_seminar_count_2025_q4": 3,
      "annual_training_hours": 18,
      "manager_professional_association": "Construction",
      "manager_site_visit_count_2025_q4": 3,
      "manager_report_delivery_channel": "Portal",
      "manager_training_format": "Classroom",
      "operations_region": "East",
      "manager_client_meeting_count_2025_q4": 10,
      "costs": {
        "P1": 3000,
        "P2": 3200,
        "P3": 3100
      }
    },
    {
      "id": "MB",
      "manager_professional_seminar_count_2025_q4": 1,
      "annual_training_hours": 18,
      "manager_professional_association": "Civil",
      "manager_site_visit_count_2025_q4": 18,
      "manager_report_delivery_channel": "Portal",
      "manager_training_format": "Classroom",
      "operations_region": "East",
      "manager_client_meeting_count_2025_q4": 4,
      "costs": {
        "P1": 2800,
        "P2": 3300,
        "P3": 2900
      }
    },
    {
      "id": "MC",
      "manager_professional_seminar_count_2025_q4": 1,
      "annual_training_hours": 12,
      "manager_professional_association": "Civil",
      "manager_site_visit_count_2025_q4": 18,
      "manager_report_delivery_channel": "Meeting",
      "manager_training_format": "Workshop",
      "operations_region": "East",
      "manager_client_meeting_count_2025_q4": 20,
      "costs": {
        "P1": 2900,
        "P2": 3100,
        "P3": 3000
      }
    }
  ],
  "projects": ["P1", "P2", "P3"]
}

##### Cost Matrix

|        | P1   | P2   | P3   |
|--------|------|------|------|
| MA     | 3000 | 3200 | 3100 |
| MB     | 2800 | 3300 | 2900 |
| MC     | 2900 | 3100 | 3000 |

##### Decision Variables

$x_{ij} = \begin{cases}
1 & \text{if manager } i \text{ is assigned to project } j \\
0 & \text{otherwise}
\end{cases}$

##### Model Summary

- Minimize total assignment cost.
- Each manager assigned to exactly one project.
- Each project assigned to exactly one manager.
- All assignments are binary decisions.
- All manager and project identifiers, costs, and manager attributes are preserved for eligibility or further constraints.