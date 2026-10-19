##### Objective Function: 

$\quad \quad \max \sum_{i \in \mathcal{C}} \text{interest\_points}_i \cdot y_i$

##### Constraints

###### 1. Credit Capacity Constraint: 

$\quad \quad \sum_{i \in \mathcal{C}} \text{credits}_i \cdot y_i \;\leq\; 20$

###### 2. Binary Decision Variables: 

$\quad \quad y_i \in \{0,1\}, \quad \forall i \in \mathcal{C}$

*Here $\mathcal{C}$ denotes the set of Operations Research courses retrieved from the data.*

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
  "credit_limit": 20
}