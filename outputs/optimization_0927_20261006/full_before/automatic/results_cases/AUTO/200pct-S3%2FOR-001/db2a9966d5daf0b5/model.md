##### Objective Function:

$\quad \min \left( 3000\,x_{\text{MA},\text{P1}} + 3200\,x_{\text{MA},\text{P2}} + 3100\,x_{\text{MA},\text{P3}} + 2800\,x_{\text{MB},\text{P1}} + 3300\,x_{\text{MB},\text{P2}} + 2900\,x_{\text{MB},\text{P3}} + 2900\,x_{\text{MC},\text{P1}} + 3100\,x_{\text{MC},\text{P2}} + 3000\,x_{\text{MC},\text{P3}} \right)$

##### Constraints

###### 1. Assignment Constraints:

$\quad x_{\text{MA},\text{P1}} + x_{\text{MA},\text{P2}} + x_{\text{MA},\text{P3}} = 1$

$\quad x_{\text{MB},\text{P1}} + x_{\text{MB},\text{P2}} + x_{\text{MB},\text{P3}} = 1$

$\quad x_{\text{MC},\text{P1}} + x_{\text{MC},\text{P2}} + x_{\text{MC},\text{P3}} = 1$

$\quad x_{\text{MA},\text{P1}} + x_{\text{MB},\text{P1}} + x_{\text{MC},\text{P1}} = 1$

$\quad x_{\text{MA},\text{P2}} + x_{\text{MB},\text{P2}} + x_{\text{MC},\text{P2}} = 1$

$\quad x_{\text{MA},\text{P3}} + x_{\text{MB},\text{P3}} + x_{\text{MC},\text{P3}} = 1$

###### 2. Variable Constraints:

$\quad x_{i,j} \in \{0,1\} \quad \forall i \in \{\text{MA}, \text{MB}, \text{MC}\},\ j \in \{\text{P1}, \text{P2}, \text{P3}\}$

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
  ]
}