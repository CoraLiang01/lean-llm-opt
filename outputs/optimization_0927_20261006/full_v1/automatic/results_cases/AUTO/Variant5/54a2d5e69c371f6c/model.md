##### Objective Function:

$\quad \min \sum_{i \in \{\text{M1},\text{M2},\text{M3},\text{M4}\}} \sum_{j \in \{\text{J1},\text{J2},\text{J3},\text{J4},\text{J5},\text{J6},\text{J7},\text{J8}\}} c_{ij} x_{ij}$

##### Constraints

###### 1. Assignment Constraints (each job assigned to exactly one team):

$\sum_{i \in \{\text{M1},\text{M2},\text{M3},\text{M4}\}} x_{ij} = 1 \quad \forall j \in \{\text{J1},\text{J2},\text{J3},\text{J4},\text{J5},\text{J6},\text{J7},\text{J8}\}$

###### 2. Capacity Constraints (team capacity not exceeded):

$\sum_{j \in \{\text{J1},\text{J2},\text{J3},\text{J4},\text{J5},\text{J6},\text{J7},\text{J8}\}} a_{ij} x_{ij} \leq b_i \quad \forall i \in \{\text{M1},\text{M2},\text{M3},\text{M4}\}$

###### 3. Binary Assignment Variables:

$x_{ij} \in \{0,1\} \quad \forall i \in \{\text{M1},\text{M2},\text{M3},\text{M4}\},\ j \in \{\text{J1},\text{J2},\text{J3},\text{J4},\text{J5},\text{J6},\text{J7},\text{J8}\}$

##### Retrieved Information

{
  "teams": ["M1", "M2", "M3", "M4"],
  "jobs": ["J1", "J2", "J3", "J4", "J5", "J6", "J7", "J8"],
  "capacity": {
    "M1": 13,
    "M2": 12,
    "M3": 12,
    "M4": 12
  },
  "cost": {
    "M1": {"J1": 8, "J2": 7, "J3": 25, "J4": 24, "J5": 27, "J6": 26, "J7": 28, "J8": 29},
    "M2": {"J1": 23, "J2": 24, "J3": 6, "J4": 9, "J5": 25, "J6": 27, "J7": 26, "J8": 28},
    "M3": {"J1": 27, "J2": 26, "J3": 24, "J4": 25, "J5": 5, "J6": 8, "J7": 23, "J8": 24},
    "M4": {"J1": 25, "J2": 27, "J3": 26, "J4": 24, "J5": 23, "J6": 25, "J7": 6, "J8": 7}
  },
  "resource": {
    "M1": {"J1": 5, "J2": 6, "J3": 8, "J4": 7, "J5": 9, "J6": 8, "J7": 7, "J8": 7},
    "M2": {"J1": 8, "J2": 7, "J3": 4, "J4": 7, "J5": 8, "J6": 9, "J7": 8, "J8": 7},
    "M3": {"J1": 9, "J2": 8, "J3": 7, "J4": 8, "J5": 6, "J6": 5, "J7": 8, "J8": 7},
    "M4": {"J1": 8, "J2": 8, "J3": 7, "J4": 8, "J5": 8, "J6": 7, "J7": 5, "J8": 6}
  }
}