##### Decision Variables:

Let $x_{ij} = \begin{cases} 1 & \text{if manager } i \text{ is assigned to project } j \\ 0 & \text{otherwise} \end{cases}$

where $i \in \{\text{MA}, \text{MB}, \text{MC}\}$ and $j \in \{\text{P1}, \text{P2}, \text{P3}\}$.

##### Parameters (from CSV):

Cost matrix $c_{ij}$:

|        | P1   | P2   | P3   |
|--------|------|------|------|
| MA     | 3000 | 3200 | 3100 |
| MB     | 2800 | 3300 | 2900 |
| MC     | 2900 | 3100 | 3000 |

##### Objective Function:

$\min \left( 3000\,x_{\text{MA},\text{P1}} + 3200\,x_{\text{MA},\text{P2}} + 3100\,x_{\text{MA},\text{P3}} + 2800\,x_{\text{MB},\text{P1}} + 3300\,x_{\text{MB},\text{P2}} + 2900\,x_{\text{MB},\text{P3}} + 2900\,x_{\text{MC},\text{P1}} + 3100\,x_{\text{MC},\text{P2}} + 3000\,x_{\text{MC},\text{P3}} \right)$

##### Constraints:

1. **Each manager is assigned to exactly one project:**

$x_{\text{MA},\text{P1}} + x_{\text{MA},\text{P2}} + x_{\text{MA},\text{P3}} = 1$

$x_{\text{MB},\text{P1}} + x_{\text{MB},\text{P2}} + x_{\text{MB},\text{P3}} = 1$

$x_{\text{MC},\text{P1}} + x_{\text{MC},\text{P2}} + x_{\text{MC},\text{P3}} = 1$

2. **Each project is assigned to exactly one manager:**

$x_{\text{MA},\text{P1}} + x_{\text{MB},\text{P1}} + x_{\text{MC},\text{P1}} = 1$

$x_{\text{MA},\text{P2}} + x_{\text{MB},\text{P2}} + x_{\text{MC},\text{P2}} = 1$

$x_{\text{MA},\text{P3}} + x_{\text{MB},\text{P3}} + x_{\text{MC},\text{P3}} = 1$

3. **Variable Domains:**

$x_{ij} \in \{0,1\}$ for all $i \in \{\text{MA}, \text{MB}, \text{MC}\}$, $j \in \{\text{P1}, \text{P2}, \text{P3}\}$

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