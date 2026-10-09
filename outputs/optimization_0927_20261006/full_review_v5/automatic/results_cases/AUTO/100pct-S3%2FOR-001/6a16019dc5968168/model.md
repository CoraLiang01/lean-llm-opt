##### Objective Function:

$\quad \min \sum_{i \in \{\text{MA}, \text{MB}, \text{MC}\}} \sum_{j \in \{\text{P1}, \text{P2}, \text{P3}\}} c_{ij} x_{ij}$

where $c_{ij}$ is the cost for manager $i$ to complete project $j$, and $x_{ij}$ is a binary variable equal to 1 if manager $i$ is assigned to project $j$, 0 otherwise.

##### Constraints

###### 1. Each manager is assigned to exactly one project:

$\sum_{j \in \{\text{P1}, \text{P2}, \text{P3}\}} x_{ij} = 1 \quad \forall i \in \{\text{MA}, \text{MB}, \text{MC}\}$

###### 2. Each project is assigned to exactly one manager:

$\sum_{i \in \{\text{MA}, \text{MB}, \text{MC}\}} x_{ij} = 1 \quad \forall j \in \{\text{P1}, \text{P2}, \text{P3}\}$

###### 3. Variable domains:

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
    "MA",
    "MB",
    "MC"
  ],
  "projects": [
    "P1",
    "P2",
    "P3"
  ],
  "additional_fields": {
    "MA": {
      "previous_period_P1": 2610,
      "two_periods_ago_assignment_status": "Reserved",
      "previous_period_P2": 3340,
      "previous_period_assignment_status": "Completed"
    },
    "MB": {
      "previous_period_P1": 2358,
      "two_periods_ago_assignment_status": "Completed",
      "previous_period_P2": 2685,
      "previous_period_assignment_status": "Completed"
    },
    "MC": {
      "previous_period_P1": 3130,
      "two_periods_ago_assignment_status": "Available",
      "previous_period_P2": 3616,
      "previous_period_assignment_status": "Available"
    }
  }
}

All cost parameters and identifiers are preserved as in the source data. Only the current period costs ("P1", "P2", "P3") are used in the objective, as per the problem description. Additional fields are included for completeness but do not affect the assignment model.