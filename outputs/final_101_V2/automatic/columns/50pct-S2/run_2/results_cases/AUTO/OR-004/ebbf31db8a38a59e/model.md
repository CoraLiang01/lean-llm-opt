##### Objective Function:

$\quad \min \sum_{i=1}^{12} \sum_{j=1}^{12} c_{ij} x_{ij}$

where $c_{ij}$ is the cost of assigning Machine $i$ to Task $j$, and $x_{ij}$ is a binary variable equal to 1 if Machine $i$ is assigned to Task $j$, 0 otherwise.

##### Constraints

###### 1. Each machine is assigned to exactly one task:

$\sum_{j=1}^{12} x_{ij} = 1 \quad \forall i \in \{1,2,\ldots,12\}$

###### 2. Each task is assigned to exactly one machine:

$\sum_{i=1}^{12} x_{ij} = 1 \quad \forall j \in \{1,2,\ldots,12\}$

###### 3. Variable constraints:

$x_{ij} \in \{0,1\} \quad \forall i,j$

##### Retrieved Information

{
  "cost": {
    "Machine 1":   {"Task 1": 167.4, "Task 2": 98.6,  "Task 3": 189.4, "Task 4": 119.6, "Task 5": 182.0, "Task 6": 145.1, "Task 7": 185.4, "Task 8": 94.8,  "Task 9": 122.3, "Task 10": 123.3, "Task 11": 96.1,  "Task 12": 90.3},
    "Machine 2":   {"Task 1": 156.2, "Task 2": 88.7,  "Task 3": 187.3, "Task 4": 124.7, "Task 5": 173.2, "Task 6": 144.3, "Task 7": 179.0, "Task 8": 91.5,  "Task 9": 115.1, "Task 10": 119.5, "Task 11": 100.1, "Task 12": 88.6},
    "Machine 3":   {"Task 1": 184.3, "Task 2": 121.0, "Task 3": 216.6, "Task 4": 140.0, "Task 5": 196.2, "Task 6": 168.8, "Task 7": 205.6, "Task 8": 114.2, "Task 9": 133.3, "Task 10": 144.5, "Task 11": 116.0, "Task 12": 107.7},
    "Machine 4":   {"Task 1": 157.9, "Task 2": 92.9,  "Task 3": 185.1, "Task 4": 120.3, "Task 5": 175.1, "Task 6": 146.2, "Task 7": 180.8, "Task 8": 86.3,  "Task 9": 111.6, "Task 10": 115.9, "Task 11": 98.1,  "Task 12": 91.1},
    "Machine 5":   {"Task 1": 175.6, "Task 2": 103.6, "Task 3": 204.5, "Task 4": 130.0, "Task 5": 192.8, "Task 6": 157.5, "Task 7": 194.2, "Task 8": 106.9, "Task 9": 129.9, "Task 10": 134.9, "Task 11": 105.8, "Task 12": 98.6},
    "Machine 6":   {"Task 1": 166.8, "Task 2": 107.0, "Task 3": 199.2, "Task 4": 130.4, "Task 5": 183.6, "Task 6": 159.5, "Task 7": 187.0, "Task 8": 98.2,  "Task 9": 121.3, "Task 10": 126.2, "Task 11": 105.9, "Task 12": 101.8},
    "Machine 7":   {"Task 1": 159.7, "Task 2": 93.2,  "Task 3": 183.8, "Task 4": 113.0, "Task 5": 171.9, "Task 6": 139.1, "Task 7": 169.6, "Task 8": 85.1,  "Task 9": 110.0, "Task 10": 116.7, "Task 11": 90.6,  "Task 12": 85.2},
    "Machine 8":   {"Task 1": 184.8, "Task 2": 115.9, "Task 3": 205.1, "Task 4": 138.6, "Task 5": 195.4, "Task 6": 160.1, "Task 7": 200.2, "Task 8": 108.5, "Task 9": 136.9, "Task 10": 140.0, "Task 11": 114.6, "Task 12": 103.9},
    "Machine 9":   {"Task 1": 157.3, "Task 2": 86.2,  "Task 3": 186.0, "Task 4": 113.9, "Task 5": 166.2, "Task 6": 136.8, "Task 7": 167.5, "Task 8": 78.8,  "Task 9": 107.4, "Task 10": 114.5, "Task 11": 87.2,  "Task 12": 78.6},
    "Machine 10":  {"Task 1": 164.8, "Task 2": 97.8,  "Task 3": 200.9, "Task 4": 125.8, "Task 5": 188.9, "Task 6": 151.2, "Task 7": 187.7, "Task 8": 99.5,  "Task 9": 119.5, "Task 10": 132.1, "Task 11": 101.1, "Task 12": 98.4},
    "Machine 11":  {"Task 1": 164.0, "Task 2": 92.2,  "Task 3": 186.2, "Task 4": 115.7, "Task 5": 174.5, "Task 6": 143.0, "Task 7": 175.9, "Task 8": 92.3,  "Task 9": 114.0, "Task 10": 121.2, "Task 11": 93.7,  "Task 12": 91.2},
    "Machine 12":  {"Task 1": 151.7, "Task 2": 76.7,  "Task 3": 179.5, "Task 4": 109.5, "Task 5": 160.6, "Task 6": 128.4, "Task 7": 170.2, "Task 8": 74.4,  "Task 9": 103.7, "Task 10": 110.4, "Task 11": 83.7,  "Task 12": 75.2}
  },
  "machines": [
    "Machine 1", "Machine 2", "Machine 3", "Machine 4", "Machine 5", "Machine 6",
    "Machine 7", "Machine 8", "Machine 9", "Machine 10", "Machine 11", "Machine 12"
  ],
  "tasks": [
    "Task 1", "Task 2", "Task 3", "Task 4", "Task 5", "Task 6",
    "Task 7", "Task 8", "Task 9", "Task 10", "Task 11", "Task 12"
  ]
}