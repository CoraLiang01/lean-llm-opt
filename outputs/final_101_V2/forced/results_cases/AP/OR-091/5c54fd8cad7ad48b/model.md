##### Objective Function:

$\quad \max \sum_{i=1}^{7} p_i x_i$

where $p_i$ is the interest points for course $i$, and $x_i \in \{0,1\}$ indicates whether course $i$ is selected.

##### Constraints

$\sum_{i=1}^{7} c_i x_i \leq 20$

where $c_i$ is the number of credits for course $i$.

$x_i \in \{0,1\}, \quad \forall i = 1, \ldots, 7$

##### Retrieved Information

{
  "courses": [
    {
      "CourseID": "C22",
      "Course Name": "Operations Research: Linear Programming",
      "Credits": 5,
      "Interest Points": 95
    },
    {
      "CourseID": "C23",
      "Course Name": "Integer Programming",
      "Credits": 5,
      "Interest Points": 92
    },
    {
      "CourseID": "C24",
      "Course Name": "Stochastic Processes",
      "Credits": 4,
      "Interest Points": 86
    },
    {
      "CourseID": "C25",
      "Course Name": "Simulation Modeling",
      "Credits": 4,
      "Interest Points": 82
    },
    {
      "CourseID": "C26",
      "Course Name": "Network Flows",
      "Credits": 4,
      "Interest Points": 85
    },
    {
      "CourseID": "C27",
      "Course Name": "Queueing Theory",
      "Credits": 4,
      "Interest Points": 80
    },
    {
      "CourseID": "C28",
      "Course Name": "Revenue Management",
      "Credits": 4,
      "Interest Points": 88
    }
  ],
  "credits": {
    "C22": 5,
    "C23": 5,
    "C24": 4,
    "C25": 4,
    "C26": 4,
    "C27": 4,
    "C28": 4
  },
  "interest_points": {
    "C22": 95,
    "C23": 92,
    "C24": 86,
    "C25": 82,
    "C26": 85,
    "C27": 80,
    "C28": 88
  }
}

##### Full Mathematical Model

Let $x_{C22}, x_{C23}, x_{C24}, x_{C25}, x_{C26}, x_{C27}, x_{C28} \in \{0,1\}$ indicate whether each course is selected.

$\max \ 95x_{C22} + 92x_{C23} + 86x_{C24} + 82x_{C25} + 85x_{C26} + 80x_{C27} + 88x_{C28}$

subject to

$5x_{C22} + 5x_{C23} + 4x_{C24} + 4x_{C25} + 4x_{C26} + 4x_{C27} + 4x_{C28} \leq 20$

$x_{C22}, x_{C23}, x_{C24}, x_{C25}, x_{C26}, x_{C27}, x_{C28} \in \{0,1\}$