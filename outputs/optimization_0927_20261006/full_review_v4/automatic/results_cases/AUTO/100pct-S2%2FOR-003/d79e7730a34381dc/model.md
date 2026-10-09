##### Objective Function:

$\quad \min \sum_{i=1}^7 \sum_{j=1}^7 c_{ij} x_{ij}$

where $c_{ij}$ is the cost for Manager $i$ to complete Project $j$, and $x_{ij}$ is a binary variable equal to 1 if Manager $i$ is assigned to Project $j$, 0 otherwise.

##### Constraints

###### 1. Assignment Constraints:

Each manager is assigned to exactly one project:
$$
\sum_{j=1}^7 x_{ij} = 1 \quad \forall i \in \{1,2,3,4,5,6,7\}
$$

Each project is assigned to exactly one manager:
$$
\sum_{i=1}^7 x_{ij} = 1 \quad \forall j \in \{1,2,3,4,5,6,7\}
$$

###### 2. Variable Constraints:

$$
x_{ij} \in \{0,1\} \quad \forall i,j
$$

##### Retrieved Information

{
  "cost": {
    "Manager 1": {
      "Project 1": 2972,
      "Project 2": 2727,
      "Project 3": 2795,
      "Project 4": 2922,
      "Project 5": 1302,
      "Project 6": 2489,
      "Project 7": 1533
    },
    "Manager 2": {
      "Project 1": 1094,
      "Project 2": 2158,
      "Project 3": 2990,
      "Project 4": 1844,
      "Project 5": 2887,
      "Project 6": 2021,
      "Project 7": 2288
    },
    "Manager 3": {
      "Project 1": 2133,
      "Project 2": 1675,
      "Project 3": 2422,
      "Project 4": 2639,
      "Project 5": 1033,
      "Project 6": 2261,
      "Project 7": 1695
    },
    "Manager 4": {
      "Project 1": 1951,
      "Project 2": 2309,
      "Project 3": 2070,
      "Project 4": 2802,
      "Project 5": 2328,
      "Project 6": 1313,
      "Project 7": 2434
    },
    "Manager 5": {
      "Project 1": 1269,
      "Project 2": 2153,
      "Project 3": 1296,
      "Project 4": 2685,
      "Project 5": 2627,
      "Project 6": 1610,
      "Project 7": 1641
    },
    "Manager 6": {
      "Project 1": 1220,
      "Project 2": 1192,
      "Project 3": 2907,
      "Project 4": 2622,
      "Project 5": 2595,
      "Project 6": 1261,
      "Project 7": 2384
    },
    "Manager 7": {
      "Project 1": 1286,
      "Project 2": 1659,
      "Project 3": 1179,
      "Project 4": 1348,
      "Project 5": 1420,
      "Project 6": 2862,
      "Project 7": 1959
    }
  },
  "managers": [
    {
      "name": "Manager 1",
      "operations_region": "South",
      "team_support_staff_count": 6,
      "manager_professional_seminar_count_2025_q4": 6,
      "manager_site_visit_count_2025_q4": 12,
      "annual_inspection_count": 3,
      "manager_safety_briefing_count_2025_q4": 6,
      "manager_professional_association": "Civil",
      "annual_training_hours": 12
    },
    {
      "name": "Manager 2",
      "operations_region": "East",
      "team_support_staff_count": 10,
      "manager_professional_seminar_count_2025_q4": 6,
      "manager_site_visit_count_2025_q4": 12,
      "annual_inspection_count": 2,
      "manager_safety_briefing_count_2025_q4": 2,
      "manager_professional_association": "Construction",
      "annual_training_hours": 36
    },
    {
      "name": "Manager 3",
      "operations_region": "North",
      "team_support_staff_count": 2,
      "manager_professional_seminar_count_2025_q4": 2,
      "manager_site_visit_count_2025_q4": 9,
      "annual_inspection_count": 6,
      "manager_safety_briefing_count_2025_q4": 12,
      "manager_professional_association": "Construction",
      "annual_training_hours": 48
    },
    {
      "name": "Manager 4",
      "operations_region": "South",
      "team_support_staff_count": 8,
      "manager_professional_seminar_count_2025_q4": 1,
      "manager_site_visit_count_2025_q4": 6,
      "annual_inspection_count": 1,
      "manager_safety_briefing_count_2025_q4": 8,
      "manager_professional_association": "Civil",
      "annual_training_hours": 48
    },
    {
      "name": "Manager 5",
      "operations_region": "North",
      "team_support_staff_count": 4,
      "manager_professional_seminar_count_2025_q4": 4,
      "manager_site_visit_count_2025_q4": 12,
      "annual_inspection_count": 3,
      "manager_safety_briefing_count_2025_q4": 12,
      "manager_professional_association": "Civil",
      "annual_training_hours": 12
    },
    {
      "name": "Manager 6",
      "operations_region": "South",
      "team_support_staff_count": 8,
      "manager_professional_seminar_count_2025_q4": 2,
      "manager_site_visit_count_2025_q4": 18,
      "annual_inspection_count": 4,
      "manager_safety_briefing_count_2025_q4": 8,
      "manager_professional_association": "General",
      "annual_training_hours": 12
    },
    {
      "name": "Manager 7",
      "operations_region": "East",
      "team_support_staff_count": 10,
      "manager_professional_seminar_count_2025_q4": 3,
      "manager_site_visit_count_2025_q4": 12,
      "annual_inspection_count": 1,
      "manager_safety_briefing_count_2025_q4": 12,
      "manager_professional_association": "Construction",
      "annual_training_hours": 12
    }
  ],
  "projects": [
    "Project 1",
    "Project 2",
    "Project 3",
    "Project 4",
    "Project 5",
    "Project 6",
    "Project 7"
  ]
}