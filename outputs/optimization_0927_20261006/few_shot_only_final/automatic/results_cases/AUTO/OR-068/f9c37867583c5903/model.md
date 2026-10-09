##### Decision Variables

Let $x_{ij} = \begin{cases} 1 & \text{if manager } i \text{ is assigned to project } j \\ 0 & \text{otherwise} \end{cases}$

where $i \in \{\text{MA}, \text{MB}, \text{MC}, \text{MD}, \text{ME}, \text{MF}\}$ and $j \in \{\text{P1}, \text{P2}, \text{P3}, \text{P4}, \text{P5}, \text{P6}\}$.

##### Parameters

The assignment costs $c_{ij}$ are as follows:

|        | P1   | P2   | P3   | P4   | P5   | P6   |
|--------|------|------|------|------|------|------|
| MA     | 2216 | 1911 | 1661 | 2122 | 1442 | 1442 |
| MB     | 1100 | 1271 | 2764 | 2557 | 1036 | 1036 |
| MC     | 2827 | 2784 | 2206 | 2216 | 2677 | 2677 |
| MD     | 2627 | 1273 | 2610 | 1957 | 1594 | 1594 |
| ME     | 3359 | 1003 | 2554 | 1706 | 2065 | 2065 |
| MF     | 1579 | 2289 | 2368 | 1922 | 2740 | 2740 |

##### Objective Function

$\min \sum_{i \in \{\text{MA}, \text{MB}, \text{MC}, \text{MD}, \text{ME}, \text{MF}\}} \sum_{j \in \{\text{P1}, \text{P2}, \text{P3}, \text{P4}, \text{P5}, \text{P6}\}} c_{ij} x_{ij}$

That is,

$\min \Bigg[ \begin{aligned}
&2216\,x_{\text{MA},\text{P1}} + 1911\,x_{\text{MA},\text{P2}} + 1661\,x_{\text{MA},\text{P3}} + 2122\,x_{\text{MA},\text{P4}} + 1442\,x_{\text{MA},\text{P5}} + 1442\,x_{\text{MA},\text{P6}} \\
+\, &1100\,x_{\text{MB},\text{P1}} + 1271\,x_{\text{MB},\text{P2}} + 2764\,x_{\text{MB},\text{P3}} + 2557\,x_{\text{MB},\text{P4}} + 1036\,x_{\text{MB},\text{P5}} + 1036\,x_{\text{MB},\text{P6}} \\
+\, &2827\,x_{\text{MC},\text{P1}} + 2784\,x_{\text{MC},\text{P2}} + 2206\,x_{\text{MC},\text{P3}} + 2216\,x_{\text{MC},\text{P4}} + 2677\,x_{\text{MC},\text{P5}} + 2677\,x_{\text{MC},\text{P6}} \\
+\, &2627\,x_{\text{MD},\text{P1}} + 1273\,x_{\text{MD},\text{P2}} + 2610\,x_{\text{MD},\text{P3}} + 1957\,x_{\text{MD},\text{P4}} + 1594\,x_{\text{MD},\text{P5}} + 1594\,x_{\text{MD},\text{P6}} \\
+\, &3359\,x_{\text{ME},\text{P1}} + 1003\,x_{\text{ME},\text{P2}} + 2554\,x_{\text{ME},\text{P3}} + 1706\,x_{\text{ME},\text{P4}} + 2065\,x_{\text{ME},\text{P5}} + 2065\,x_{\text{ME},\text{P6}} \\
+\, &1579\,x_{\text{MF},\text{P1}} + 2289\,x_{\text{MF},\text{P2}} + 2368\,x_{\text{MF},\text{P3}} + 1922\,x_{\text{MF},\text{P4}} + 2740\,x_{\text{MF},\text{P5}} + 2740\,x_{\text{MF},\text{P6}}
\end{aligned} \Bigg]$

##### Constraints

###### 1. Each manager is assigned to exactly one project:

$\sum_{j \in \{\text{P1}, \text{P2}, \text{P3}, \text{P4}, \text{P5}, \text{P6}\}} x_{i j} = 1 \quad \forall i \in \{\text{MA}, \text{MB}, \text{MC}, \text{MD}, \text{ME}, \text{MF}\}$

###### 2. Each project is assigned to exactly one manager:

$\sum_{i \in \{\text{MA}, \text{MB}, \text{MC}, \text{MD}, \text{ME}, \text{MF}\}} x_{i j} = 1 \quad \forall j \in \{\text{P1}, \text{P2}, \text{P3}, \text{P4}, \text{P5}, \text{P6}\}$

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