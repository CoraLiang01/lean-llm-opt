##### Decision Variables:

Let  
$x_{ij} = \begin{cases} 1 & \text{if manager } i \text{ is assigned to project } j \\ 0 & \text{otherwise} \end{cases}$  
for all $i \in \{\text{MA}, \text{MB}, \text{MC}, \text{MD}, \text{ME}, \text{MF}\}$ and $j \in \{\text{P1}, \text{P2}, \text{P3}, \text{P4}, \text{P5}, \text{P6}\}$.

##### Parameters:

The assignment costs $c_{ij}$ are given as follows:

|        | P1   | P2   | P3   | P4   | P5   | P6   |
|--------|------|------|------|------|------|------|
| MA     | 2216 | 1911 | 1661 | 2122 | 1442 | 1442 |
| MB     | 1100 | 1271 | 2764 | 2557 | 1036 | 1036 |
| MC     | 2827 | 2784 | 2206 | 2216 | 2677 | 2677 |
| MD     | 2627 | 1273 | 2610 | 1957 | 1594 | 1594 |
| ME     | 3359 | 1003 | 2554 | 1706 | 2065 | 2065 |
| MF     | 1579 | 2289 | 2368 | 1922 | 2740 | 2740 |

##### Objective Function:

$\min \sum_{i \in \{\text{MA}, \text{MB}, \text{MC}, \text{MD}, \text{ME}, \text{MF}\}} \sum_{j \in \{\text{P1}, \text{P2}, \text{P3}, \text{P4}, \text{P5}, \text{P6}\}} c_{ij} x_{ij}$

That is,

$\min \Bigg( 2216x_{\text{MA},\text{P1}} + 1911x_{\text{MA},\text{P2}} + 1661x_{\text{MA},\text{P3}} + 2122x_{\text{MA},\text{P4}} + 1442x_{\text{MA},\text{P5}} + 1442x_{\text{MA},\text{P6}}$  
$\quad + 1100x_{\text{MB},\text{P1}} + 1271x_{\text{MB},\text{P2}} + 2764x_{\text{MB},\text{P3}} + 2557x_{\text{MB},\text{P4}} + 1036x_{\text{MB},\text{P5}} + 1036x_{\text{MB},\text{P6}}$  
$\quad + 2827x_{\text{MC},\text{P1}} + 2784x_{\text{MC},\text{P2}} + 2206x_{\text{MC},\text{P3}} + 2216x_{\text{MC},\text{P4}} + 2677x_{\text{MC},\text{P5}} + 2677x_{\text{MC},\text{P6}}$  
$\quad + 2627x_{\text{MD},\text{P1}} + 1273x_{\text{MD},\text{P2}} + 2610x_{\text{MD},\text{P3}} + 1957x_{\text{MD},\text{P4}} + 1594x_{\text{MD},\text{P5}} + 1594x_{\text{MD},\text{P6}}$  
$\quad + 3359x_{\text{ME},\text{P1}} + 1003x_{\text{ME},\text{P2}} + 2554x_{\text{ME},\text{P3}} + 1706x_{\text{ME},\text{P4}} + 2065x_{\text{ME},\text{P5}} + 2065x_{\text{ME},\text{P6}}$  
$\quad + 1579x_{\text{MF},\text{P1}} + 2289x_{\text{MF},\text{P2}} + 2368x_{\text{MF},\text{P3}} + 1922x_{\text{MF},\text{P4}} + 2740x_{\text{MF},\text{P5}} + 2740x_{\text{MF},\text{P6}} \Bigg)$

##### Constraints:

###### 1. Each manager is assigned to exactly one project:

$\sum_{j \in \{\text{P1}, \text{P2}, \text{P3}, \text{P4}, \text{P5}, \text{P6}\}} x_{ij} = 1 \quad \forall i \in \{\text{MA}, \text{MB}, \text{MC}, \text{MD}, \text{ME}, \text{MF}\}$

###### 2. Each project is assigned to exactly one manager:

$\sum_{i \in \{\text{MA}, \text{MB}, \text{MC}, \text{MD}, \text{ME}, \text{MF}\}} x_{ij} = 1 \quad \forall j \in \{\text{P1}, \text{P2}, \text{P3}, \text{P4}, \text{P5}, \text{P6}\}$

###### 3. Variable domains:

$x_{ij} \in \{0,1\} \quad \forall i, j$

##### Retrieved Information

{
  "cost": {
    "MA": {"P1": 2216, "P2": 1911, "P3": 1661, "P4": 2122, "P5": 1442, "P6": 1442},
    "MB": {"P1": 1100, "P2": 1271, "P3": 2764, "P4": 2557, "P5": 1036, "P6": 1036},
    "MC": {"P1": 2827, "P2": 2784, "P3": 2206, "P4": 2216, "P5": 2677, "P6": 2677},
    "MD": {"P1": 2627, "P2": 1273, "P3": 2610, "P4": 1957, "P5": 1594, "P6": 1594},
    "ME": {"P1": 3359, "P2": 1003, "P3": 2554, "P4": 1706, "P5": 2065, "P6": 2065},
    "MF": {"P1": 1579, "P2": 2289, "P3": 2368, "P4": 1922, "P5": 2740, "P6": 2740}
  },
  "managers": ["MA", "MB", "MC", "MD", "ME", "MF"],
  "projects": ["P1", "P2", "P3", "P4", "P5", "P6"]
}