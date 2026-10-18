##### Decision Variables:

Let $x_i \in \{0,1\}$ for each Operations Research course $i$, where $x_i = 1$ if course $i$ is selected, $0$ otherwise.

##### Parameters:

- Let $C_i$ be the number of credits for course $i$.
- Let $P_i$ be the interest points for course $i$.

The set of Operations Research courses and their parameters are:

| Course ID | Course Name                                 | $C_i$ (Credits) | $P_i$ (Interest Points) |
|-----------|---------------------------------------------|-----------------|------------------------|
| C22       | Operations Research: Linear Programming     | 5               | 95                     |
| C23       | Integer Programming                        | 5               | 92                     |
| C24       | Stochastic Processes                       | 4               | 86                     |
| C25       | Simulation Modeling                        | 4               | 82                     |
| C26       | Network Flows                              | 4               | 85                     |
| C27       | Queueing Theory                            | 4               | 80                     |
| C28       | Revenue Management                         | 4               | 88                     |

##### Objective Function:

$\quad \max \left(95x_{22} + 92x_{23} + 86x_{24} + 82x_{25} + 85x_{26} + 80x_{27} + 88x_{28}\right)$

##### Constraints:

$\quad 5x_{22} + 5x_{23} + 4x_{24} + 4x_{25} + 4x_{26} + 4x_{27} + 4x_{28} \leq 20$

$\quad x_{22}, x_{23}, x_{24}, x_{25}, x_{26}, x_{27}, x_{28} \in \{0,1\}$

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