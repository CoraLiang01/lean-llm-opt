##### Objective Function:

$\quad \min \sum_{i \in \{\text{MA}, \text{MB}, \text{MC}, \text{MD}, \text{ME}, \text{MF}\}} \sum_{j \in \{\text{P1}, \text{P2}, \text{P3}, \text{P4}, \text{P5}, \text{P6}\}} c_{ij} x_{ij}$

where $c_{ij}$ is the cost of assigning manager $i$ to project $j$, and $x_{ij}$ is a binary variable equal to 1 if manager $i$ is assigned to project $j$, 0 otherwise.

##### Constraints

###### 1. Assignment Constraints:

Each manager is assigned to exactly one project:
$$
\sum_{j \in \{\text{P1}, \text{P2}, \text{P3}, \text{P4}, \text{P5}, \text{P6}\}} x_{ij} = 1 \quad \forall i \in \{\text{MA}, \text{MB}, \text{MC}, \text{MD}, \text{ME}, \text{MF}\}
$$

Each project is assigned to exactly one manager:
$$
\sum_{i \in \{\text{MA}, \text{MB}, \text{MC}, \text{MD}, \text{ME}, \text{MF}\}} x_{ij} = 1 \quad \forall j \in \{\text{P1}, \text{P2}, \text{P3}, \text{P4}, \text{P5}, \text{P6}\}
$$

###### 2. Variable Constraints:

$$
x_{ij} \in \{0,1\} \quad \forall i \in \{\text{MA}, \text{MB}, \text{MC}, \text{MD}, \text{ME}, \text{MF}\},\ j \in \{\text{P1}, \text{P2}, \text{P3}, \text{P4}, \text{P5}, \text{P6}\}
$$

##### Retrieved Information

{
  "cost": {
    "MA": {
      "P1": 2216,
      "P2": 1911,
      "P3": 1661,
      "P4": 2122,
      "P5": 1442,
      "P6": 1442
    },
    "MB": {
      "P1": 1100,
      "P2": 1271,
      "P3": 2764,
      "P4": 2557,
      "P5": 1036,
      "P6": 1036
    },
    "MC": {
      "P1": 2827,
      "P2": 2784,
      "P3": 2206,
      "P4": 2216,
      "P5": 2677,
      "P6": 2677
    },
    "MD": {
      "P1": 2627,
      "P2": 1273,
      "P3": 2610,
      "P4": 1957,
      "P5": 1594,
      "P6": 1594
    },
    "ME": {
      "P1": 3359,
      "P2": 1003,
      "P3": 2554,
      "P4": 1706,
      "P5": 2065,
      "P6": 2065
    },
    "MF": {
      "P1": 1579,
      "P2": 2289,
      "P3": 2368,
      "P4": 1922,
      "P5": 2740,
      "P6": 2740
    }
  },
  "managers": [
    "MA",
    "MB",
    "MC",
    "MD",
    "ME",
    "MF"
  ],
  "projects": [
    "P1",
    "P2",
    "P3",
    "P4",
    "P5",
    "P6"
  ],
  "manager_attributes": {
    "MA": {
      "team_support_staff_count": 6,
      "annual_training_hours": 24,
      "manager_site_visit_count_2025_q4": 12,
      "annual_inspection_count": 4,
      "manager_safety_briefing_count_2025_q4": 12,
      "manager_training_format": "Workshop",
      "manager_design_review_count_2025_q4": 10,
      "manager_mentoring_session_count_2025_q4": 1,
      "manager_client_meeting_count_2025_q4": 10,
      "manager_professional_seminar_count_2025_q4": 1,
      "manager_report_delivery_channel": "Email",
      "operations_region": "West",
      "manager_professional_association": "General",
      "manager_procurement_inquiry_count_2025_q4": 3
    },
    "MB": {
      "team_support_staff_count": 6,
      "annual_training_hours": 18,
      "manager_site_visit_count_2025_q4": 12,
      "annual_inspection_count": 4,
      "manager_safety_briefing_count_2025_q4": 12,
      "manager_training_format": "Workshop",
      "manager_design_review_count_2025_q4": 7,
      "manager_mentoring_session_count_2025_q4": 8,
      "manager_client_meeting_count_2025_q4": 4,
      "manager_professional_seminar_count_2025_q4": 4,
      "manager_report_delivery_channel": "Portal",
      "operations_region": "East",
      "manager_professional_association": "General",
      "manager_procurement_inquiry_count_2025_q4": 3
    },
    "MC": {
      "team_support_staff_count": 4,
      "annual_training_hours": 36,
      "manager_site_visit_count_2025_q4": 9,
      "annual_inspection_count": 6,
      "manager_safety_briefing_count_2025_q4": 8,
      "manager_training_format": "Workshop",
      "manager_design_review_count_2025_q4": 7,
      "manager_mentoring_session_count_2025_q4": 4,
      "manager_client_meeting_count_2025_q4": 7,
      "manager_professional_seminar_count_2025_q4": 2,
      "manager_report_delivery_channel": "Portal",
      "operations_region": "East",
      "manager_professional_association": "General",
      "manager_procurement_inquiry_count_2025_q4": 3
    },
    "MD": {
      "team_support_staff_count": 2,
      "annual_training_hours": 48,
      "manager_site_visit_count_2025_q4": 6,
      "annual_inspection_count": 2,
      "manager_safety_briefing_count_2025_q4": 2,
      "manager_training_format": "Online",
      "manager_design_review_count_2025_q4": 7,
      "manager_mentoring_session_count_2025_q4": 6,
      "manager_client_meeting_count_2025_q4": 20,
      "manager_professional_seminar_count_2025_q4": 2,
      "manager_report_delivery_channel": "Email",
      "operations_region": "West",
      "manager_professional_association": "Civil",
      "manager_procurement_inquiry_count_2025_q4": 8
    },
    "ME": {
      "team_support_staff_count": 6,
      "annual_training_hours": 18,
      "manager_site_visit_count_2025_q4": 6,
      "annual_inspection_count": 1,
      "manager_safety_briefing_count_2025_q4": 12,
      "manager_training_format": "Online",
      "manager_design_review_count_2025_q4": 7,
      "manager_mentoring_session_count_2025_q4": 2,
      "manager_client_meeting_count_2025_q4": 20,
      "manager_professional_seminar_count_2025_q4": 2,
      "manager_report_delivery_channel": "Portal",
      "operations_region": "East",
      "manager_professional_association": "General",
      "manager_procurement_inquiry_count_2025_q4": 8
    },
    "MF": {
      "team_support_staff_count": 10,
      "annual_training_hours": 48,
      "manager_site_visit_count_2025_q4": 12,
      "annual_inspection_count": 1,
      "manager_safety_briefing_count_2025_q4": 8,
      "manager_training_format": "Workshop",
      "manager_design_review_count_2025_q4": 2,
      "manager_mentoring_session_count_2025_q4": 6,
      "manager_client_meeting_count_2025_q4": 20,
      "manager_professional_seminar_count_2025_q4": 2,
      "manager_report_delivery_channel": "Portal",
      "operations_region": "East",
      "manager_professional_association": "Civil",
      "manager_procurement_inquiry_count_2025_q4": 8
    }
  }
}