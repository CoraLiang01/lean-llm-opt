##### Decision Variables:

Let $x_i$ be a binary variable for each Operations Research course $i$, where:
- $x_i = 1$ if course $i$ is selected,
- $x_i = 0$ otherwise.

##### Parameters:

Let the set of Operations Research courses be:
- $C = \{\text{C22}, \text{C23}, \text{C24}, \text{C25}, \text{C26}, \text{C27}, \text{C28}\}$

For each course $i \in C$:

| Course ID | Course Name                              | Credits | Interest Points |
|-----------|------------------------------------------|---------|-----------------|
| C22       | Operations Research: Linear Programming  | 5       | 95              |
| C23       | Integer Programming                      | 5       | 92              |
| C24       | Stochastic Processes                     | 4       | 86              |
| C25       | Simulation Modeling                      | 4       | 82              |
| C26       | Network Flows                            | 4       | 85              |
| C27       | Queueing Theory                          | 4       | 80              |
| C28       | Revenue Management                       | 4       | 88              |

Let $a_i$ be the number of credits for course $i$.

Let $p_i$ be the interest points for course $i$.

##### Objective Function:

$\quad \max \sum_{i \in C} p_i x_i$

##### Constraints:

$\sum_{i \in C} a_i x_i \leq 20$

$x_i \in \{0,1\} \quad \forall i \in C$

##### Retrieved Information

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
  ]
}