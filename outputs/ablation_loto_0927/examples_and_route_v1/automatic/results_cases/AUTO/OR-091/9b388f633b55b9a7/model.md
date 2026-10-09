Let $x_i$ be a binary variable indicating whether Operations Research course $i$ is selected ($x_i = 1$) or not ($x_i = 0$).

Let the set of Operations Research courses and their parameters be:

| course_id | course_name                              | credits | interest_points |
|-----------|------------------------------------------|---------|-----------------|
| C22       | Operations Research: Linear Programming  | 5       | 95              |
| C23       | Integer Programming                      | 5       | 92              |
| C24       | Stochastic Processes                     | 4       | 86              |
| C25       | Simulation Modeling                      | 4       | 82              |
| C26       | Network Flows                            | 4       | 85              |
| C27       | Queueing Theory                          | 4       | 80              |
| C28       | Revenue Management                       | 4       | 88              |

Define decision variables:
$$
x_{C22},\ x_{C23},\ x_{C24},\ x_{C25},\ x_{C26},\ x_{C27},\ x_{C28} \in \{0,1\}
$$

Objective:
$$
\max\ 95x_{C22} + 92x_{C23} + 86x_{C24} + 82x_{C25} + 85x_{C26} + 80x_{C27} + 88x_{C28}
$$

Subject to:
$$
5x_{C22} + 5x_{C23} + 4x_{C24} + 4x_{C25} + 4x_{C26} + 4x_{C27} + 4x_{C28} \leq 20
$$

$$
x_{C22},\ x_{C23},\ x_{C24},\ x_{C25},\ x_{C26},\ x_{C27},\ x_{C28} \in \{0,1\}
$$

Where each $x_i$ indicates whether to enroll in course $i$ (1) or not (0). The objective is to maximize total interest points from selected Operations Research courses, subject to a total credit limit of 20.