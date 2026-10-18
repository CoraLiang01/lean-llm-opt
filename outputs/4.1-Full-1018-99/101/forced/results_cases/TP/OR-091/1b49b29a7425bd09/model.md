##### Decision Variables

For each Operations Research course $i\in\{C22, C23, C24, C25, C26, C27, C28\}$:

$y_i\in\{0,1\}$: $1$ if course $i$ is selected, $0$ otherwise.

##### Parameters

\[
\begin{array}{llll}
\text{Course ID} & \text{Course Name} & \text{Credits} & \text{Interest Points} \\
\hline
C22 & \text{Operations Research: Linear Programming} & 5 & 95 \\
C23 & \text{Integer Programming} & 5 & 92 \\
C24 & \text{Stochastic Processes} & 4 & 86 \\
C25 & \text{Simulation Modeling} & 4 & 82 \\
C26 & \text{Network Flows} & 4 & 85 \\
C27 & \text{Queueing Theory} & 4 & 80 \\
C28 & \text{Revenue Management} & 4 & 88 \\
\end{array}
\]

##### Objective Function

\[
\max\ 95y_{C22} + 92y_{C23} + 86y_{C24} + 82y_{C25} + 85y_{C26} + 80y_{C27} + 88y_{C28}
\]

##### Constraints

\[
5y_{C22} + 5y_{C23} + 4y_{C24} + 4y_{C25} + 4y_{C26} + 4y_{C27} + 4y_{C28} \leq 20
\]
\[
y_i \in \{0,1\} \quad \forall i \in \{C22, C23, C24, C25, C26, C27, C28\}
\]

###### Retrieved Information

{
  "courses": [
    {"course_id": "C22", "course_name": "Operations Research: Linear Programming", "credits": 5, "interest_points": 95},
    {"course_id": "C23", "course_name": "Integer Programming", "credits": 5, "interest_points": 92},
    {"course_id": "C24", "course_name": "Stochastic Processes", "credits": 4, "interest_points": 86},
    {"course_id": "C25", "course_name": "Simulation Modeling", "credits": 4, "interest_points": 82},
    {"course_id": "C26", "course_name": "Network Flows", "credits": 4, "interest_points": 85},
    {"course_id": "C27", "course_name": "Queueing Theory", "credits": 4, "interest_points": 80},
    {"course_id": "C28", "course_name": "Revenue Management", "credits": 4, "interest_points": 88}
  ]
}