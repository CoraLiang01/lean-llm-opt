##### Objective Function:

$\quad \min \sum_{i \in \{\text{MA}, \text{MB}, \text{MC}\}} \sum_{j \in \{\text{P1}, \text{P2}, \text{P3}\}} c_{ij} x_{ij}$

##### Constraints

###### 1. Assignment Constraints:

$\sum_{j \in \{\text{P1}, \text{P2}, \text{P3}\}} x_{ij} = 1 \quad \forall i \in \{\text{MA}, \text{MB}, \text{MC}\}$

$\sum_{i \in \{\text{MA}, \text{MB}, \text{MC}\}} x_{ij} = 1 \quad \forall j \in \{\text{P1}, \text{P2}, \text{P3}\}$

###### 2. Variable Constraints:

$x_{ij} \in \{0,1\} \quad \forall i \in \{\text{MA}, \text{MB}, \text{MC}\},\ j \in \{\text{P1}, \text{P2}, \text{P3}\}$

##### Retrieved Information

Managers and their attributes:
- MA: 
  - manager_professional_seminar_count_2025_q4: 3
  - annual_training_hours: 18
  - manager_professional_association: Construction
  - manager_site_visit_count_2025_q4: 3
  - manager_report_delivery_channel: Portal
  - manager_training_format: Classroom
  - operations_region: East
  - manager_client_meeting_count_2025_q4: 10
- MB:
  - manager_professional_seminar_count_2025_q4: 1
  - annual_training_hours: 18
  - manager_professional_association: Civil
  - manager_site_visit_count_2025_q4: 18
  - manager_report_delivery_channel: Portal
  - manager_training_format: Classroom
  - operations_region: East
  - manager_client_meeting_count_2025_q4: 4
- MC:
  - manager_professional_seminar_count_2025_q4: 1
  - annual_training_hours: 12
  - manager_professional_association: Civil
  - manager_site_visit_count_2025_q4: 18
  - manager_report_delivery_channel: Meeting
  - manager_training_format: Workshop
  - operations_region: East
  - manager_client_meeting_count_2025_q4: 20

Projects:
- P1
- P2
- P3

Cost matrix $c_{ij}$:
|        | P1   | P2   | P3   |
|--------|------|------|------|
| MA     | 3000 | 3200 | 3100 |
| MB     | 2800 | 3300 | 2900 |
| MC     | 2900 | 3100 | 3000 |

Decision variables:
- $x_{ij} = 1$ if manager $i$ is assigned to project $j$, $0$ otherwise.

All managers and projects must be assigned in a one-to-one fashion to minimize total cost.