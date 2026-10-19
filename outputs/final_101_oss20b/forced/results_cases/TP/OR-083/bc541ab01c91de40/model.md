##### Decision Variables

$x_{ij}\in\{0,1\}$: 1 if worker $i\in I$ is assigned to task $j\in J$, 0 otherwise.

##### Objective Function

$\min\sum_{i\in I}\sum_{j\in J}t_{ij}x_{ij}$

##### Constraints

1. Each task is assigned to exactly one worker:  
$\sum_{i\in I}x_{ij}=1,\quad\forall j\in J$

2. Each worker is assigned to at most one task:  
$\sum_{j\in J}x_{ij}\leq1,\quad\forall i\in I$

3. Binary domain:  
$x_{ij}\in\{0,1\},\quad\forall i\in I,\forall j\in J$

$I=\{W1,W2,W3,W4,W5,W6,W7,W8,W9,W10,W11,W12\}$  
$J=\{A,B,C,D,E,F,G,H,I,J\}$

##### Retrieved Information

```json
{
  "workers": [
    "W1",
    "W2",
    "W3",
    "W4",
    "W5",
    "W6",
    "W7",
    "W8",
    "W9",
    "W10",
    "W11",
    "W12"
  ],
  "tasks": [
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "J"
  ],
  "time": {
    "W1": {
      "A": 9,
      "B": 4,
      "C": 3,
      "D": 7,
      "E": 6,
      "F": 5,
      "G": 6,
      "H": 3,
      "I": 7,
      "J": 5
    },
    "W2": {
      "A": 4,
      "B": 6,
      "C": 5,
      "D": 6,
      "E": 4,
      "F": 5,
      "G": 3,
      "H": 8,
      "I": 7,
      "J": 6
    },
    "W3": {
      "A": 5,
      "B": 4,
      "C": 7,
      "D": 5,
      "E": 6,
      "F": 6,
      "G": 5,
      "H": 8,
      "I": 6,
      "J": 9
    },
    "W4": {
      "A": 7,
      "B": 5,
      "C": 2,
      "D": 3,
      "E": 7,
      "F": 8,
      "G": 5,
      "H": 6,
      "I": 8,
      "J": 5
    },
    "W5": {
      "A": 10,
      "B": 6,
      "C": 7,
      "D": 4,
      "E": 5,
      "F": 4,
      "G": 4,
      "H": 5,
      "I": 9,
      "J": 7
    },
    "W6": {
      "A": 6,
      "B": 7,
      "C": 6,
      "D": 3,
      "E": 9,
      "F": 5,
      "G": 7,
      "H": 4,
      "I": 3,
      "J": 4
    },
    "W7": {
      "A": 8,
      "B": 8,
      "C": 5,
      "D": 9,
      "E": 5,
      "F": 7,
      "G": 5,
      "H": 9,
      "I": 5,
      "J": 3
    },
    "W8": {
      "A": 7,
      "B": 4,
      "C": 8,
      "D": 8,
      "E": 6,
      "F": 7,
      "G": 5,
      "H": 7,
      "I": 7,
      "J": 7
    },
    "W9": {
      "A": 5,
      "B": 6,
      "C": 8,
      "D": 7,
      "E": 7,
      "F": 8,
      "G": 7,
      "H": 8,
      "I": 4,
      "J": 5
    },
    "W10": {
      "A": 8,
      "B": 7,
      "C": 9,
      "D": 5,
      "E": 8,
      "F": 5,
      "G": 9,
      "H": 9,
      "I": 3,
      "J": 4
    },
    "W11": {
      "A": 9,
      "B": 8,
      "C": 10,
      "D": 8,
      "E": 5,
      "F": 4,
      "G": 7,
      "H": 6,
      "I": 8,
      "J": 7
    },
    "W12": {
      "A": 8,
      "B": 5,
      "C": 6,
      "D": 9,
      "E": 4,
      "F": 7,
      "G": 8,
      "H": 4,
      "I": 7,
      "J": 9
    }
  }
}
```