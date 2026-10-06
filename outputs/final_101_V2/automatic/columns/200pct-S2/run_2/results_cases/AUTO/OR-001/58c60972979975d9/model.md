##### Objective Function:

$\quad \min \sum_{i \in \{\text{MA}, \text{MB}, \text{MC}\}} \sum_{j \in \{\text{P1}, \text{P2}, \text{P3}\}} c_{ij} x_{ij}$

where $c_{ij}$ is the cost for manager $i$ to manage project $j$, and $x_{ij}$ is a binary variable equal to 1 if manager $i$ is assigned to project $j$, 0 otherwise.

##### Constraints

###### 1. Each manager is assigned to exactly one project:

$\sum_{j \in \{\text{P1}, \text{P2}, \text{P3}\}} x_{ij} = 1 \quad \forall i \in \{\text{MA}, \text{MB}, \text{MC}\}$

###### 2. Each project is assigned to exactly one manager:

$\sum_{i \in \{\text{MA}, \text{MB}, \text{MC}\}} x_{ij} = 1 \quad \forall j \in \{\text{P1}, \text{P2}, \text{P3}\}$

###### 3. Variable Constraints:

$x_{ij} \in \{0,1\} \quad \forall i \in \{\text{MA}, \text{MB}, \text{MC}\},\ j \in \{\text{P1}, \text{P2}, \text{P3}\}$

##### Retrieved Information

{
  "cost": {
    "MA": {
      "P1": 3000,
      "P2": 3200,
      "P3": 3100
    },
    "MB": {
      "P1": 2800,
      "P2": 3300,
      "P3": 2900
    },
    "MC": {
      "P1": 2900,
      "P2": 3100,
      "P3": 3000
    }
  },
  "managers": [
    {
      "Manager": "MA",
      "Operations Region": "East",
      "Manager Professional Association": "Construction",
      "Annual Training Hours": 18,
      "Manager Training Format": "Classroom",
      "Manager Professional Seminar Count 2025 Q4": 3,
      "Manager Site Visit Count 2025 Q4": 3,
      "Manager Client Meeting Count 2025 Q4": 10,
      "Manager Report Delivery Channel": "Portal"
    },
    {
      "Manager": "MB",
      "Operations Region": "East",
      "Manager Professional Association": "Civil",
      "Annual Training Hours": 18,
      "Manager Training Format": "Classroom",
      "Manager Professional Seminar Count 2025 Q4": 1,
      "Manager Site Visit Count 2025 Q4": 18,
      "Manager Client Meeting Count 2025 Q4": 4,
      "Manager Report Delivery Channel": "Portal"
    },
    {
      "Manager": "MC",
      "Operations Region": "East",
      "Manager Professional Association": "Civil",
      "Annual Training Hours": 12,
      "Manager Training Format": "Workshop",
      "Manager Professional Seminar Count 2025 Q4": 1,
      "Manager Site Visit Count 2025 Q4": 18,
      "Manager Client Meeting Count 2025 Q4": 20,
      "Manager Report Delivery Channel": "Meeting"
    }
  ],
  "projects": [
    "P1",
    "P2",
    "P3"
  ]
}