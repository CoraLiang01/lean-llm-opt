##### Decision Variables

- $x_{ij} \geq 0$: Quantity of goods shipped from warehouse (facility) $i \in I$ to musician/band (customer) $j \in J$ (continuous).
- $y_i \in \{0,1\}$: 1 if warehouse $i$ is activated (operational), 0 otherwise (binary).

##### Parameters

- $I = \{\text{S1}, \text{S2}, \text{S3}\}$ (warehouses)
- $J = \{\text{C1}, \text{C2}, \text{C3}\}$ (musicians/bands)
- Demands:
  - $d_{\text{C1}} = 1083$
  - $d_{\text{C2}} = 776$
  - $d_{\text{C3}} = 16214$
- Fixed costs:
  - $f_{\text{S1}} = 102.33$
  - $f_{\text{S2}} = 94.92$
  - $f_{\text{S3}} = 91.83$
- Transportation costs $c_{ij}$:

|        | C1      | C2      | C3      |
|--------|---------|---------|---------|
| S1     | 1506.22 | 70.9    | 8.44    |
| S2     | 1732.65 | 1780.72 | 567.44  |
| S3     | 115.66  | 100.76  | 64.68   |

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
\]

##### Constraints

1. **Demand satisfaction:**  
   For each musician/band $j \in J$,
   \[
   \sum_{i \in I} x_{ij} = d_j
   \]
2. **Warehouse activation:**  
   For each warehouse $i \in I$,
   \[
   \sum_{j \in J} x_{ij} \leq M y_i
   \]
   where $M = \sum_{j \in J} d_j = 1083 + 776 + 16214 = 18073$ (since there are no explicit warehouse capacities).
3. **Variable domains:**  
   \[
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   \]
   \[
   y_i \in \{0,1\} \quad \forall i \in I
   \]

##### Complete Mathematical Model

\[
\begin{align*}
\min\quad & \sum_{i \in \{\text{S1}, \text{S2}, \text{S3}\}} \sum_{j \in \{\text{C1}, \text{C2}, \text{C3}\}} c_{ij} x_{ij} + \sum_{i \in \{\text{S1}, \text{S2}, \text{S3}\}} f_i y_i \\
\text{s.t.}\quad & \sum_{i \in \{\text{S1}, \text{S2}, \text{S3}\}} x_{ij} = d_j \quad \forall j \in \{\text{C1}, \text{C2}, \text{C3}\} \\
& \sum_{j \in \{\text{C1}, \text{C2}, \text{C3}\}} x_{ij} \leq 18073\, y_i \quad \forall i \in \{\text{S1}, \text{S2}, \text{S3}\} \\
& x_{ij} \geq 0 \quad \forall i, j \\
& y_i \in \{0,1\} \quad \forall i
\end{align*}
\]

Where:

- $c_{ij}$, $f_i$, and $d_j$ are as specified in the tables above.
- $M = 18073$ is the total demand, serving as a valid upper bound for the big-M constraint.

###### Retrieved Information

{
  "warehouses": [
    {"id": "S1", "fixed_cost": 102.33, "source_row": 4},
    {"id": "S2", "fixed_cost": 94.92, "source_row": 5},
    {"id": "S3", "fixed_cost": 91.83, "source_row": 6}
  ],
  "musicians_bands": [
    {"id": "C1", "demand": 1083, "source_row": 1},
    {"id": "C2", "demand": 776, "source_row": 2},
    {"id": "C3", "demand": 16214, "source_row": 3}
  ],
  "transportation_costs": {
    "S1": {"C1": 1506.22, "C2": 70.9, "C3": 8.44},
    "S2": {"C1": 1732.65, "C2": 1780.72, "C3": 567.44},
    "S3": {"C1": 115.66, "C2": 100.76, "C3": 64.68}
  },
  "M": 18073
}