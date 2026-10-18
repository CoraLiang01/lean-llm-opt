##### Decision Variables

$x_i \in \{0,1\}$: whether course $i$ is selected.

##### Objective Function

$\max \sum_{i\in I} p_i x_i$

##### Constraints

1. Credit limit: $\sum_{i\in I} c_i x_i \leq 20$.

2. Domains: $x_i \in \{0,1\}$, $\forall i\in I$.

$I=\{C22,C23,C24,C25,C26,C27,C28\}$.

###### Retrieved Information

{
  "courses": [
    {
      "course_id": "C22",
      "course_name": "Operations Research: Linear Programming",
      "credits": 5,
      "interest_points": 95
    },
    {
      "course_id": "C23",
      "course_name": "Integer Programming",
      "credits": 5,
      "interest_points": 92
    },
    {
      "course_id": "C24",
      "course_name": "Stochastic Processes",
      "credits": 4,
      "interest_points": 86
    },
    {
      "course_id": "C25",
      "course_name": "Simulation Modeling",
      "credits": 4,
      "interest_points": 82
    },
    {
      "course_id": "C26",
      "course_name": "Network Flows",
      "credits": 4,
      "interest_points": 85
    },
    {
      "course_id": "C27",
      "course_name": "Queueing Theory",
      "credits": 4,
      "interest_points": 80
    },
    {
      "course_id": "C28",
      "course_name": "Revenue Management",
      "credits": 4,
      "interest_points": 88
    }
  ],
  "credits_vector": [5,5,4,4,4,4,4],
  "interest_points_vector": [95,92,86,82,85,80,88]
}