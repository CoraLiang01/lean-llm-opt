##### Decision Variables

$x_{ij} \geq 0$: quantity shipped from warehouse $i \in I$ to store $j \in J$ (continuous).

##### Objective Function

$\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}$

##### Constraints

1. Store demand satisfaction: $\sum_{i \in I} x_{ij} = d_j,\quad \forall j \in J$
2. Warehouse supply capacity: $\sum_{j \in J} x_{ij} \leq s_i,\quad \forall i \in I$
3. Nonnegativity: $x_{ij} \geq 0,\quad \forall i \in I,\, j \in J$

Where:

- $I = \{S1, S2, S3, S4, S5\}$ (warehouses)
- $J = \{D1, D2, D3, D4, D5\}$ (stores)

##### Parameters

- Store demands ($d_j$):

  - $d_{D1} = 428$
  - $d_{D2} = 217$
  - $d_{D3} = 214$
  - $d_{D4} = 380$
  - $d_{D5} = 254$

- Warehouse supply capacities ($s_i$):

  - $s_{S1} = 428$
  - $s_{S2} = 217$
  - $s_{S3} = 214$
  - $s_{S4} = 380$
  - $s_{S5} = 254$

- Transportation costs per unit ($c_{ij}$):

|        | D1              | D2              | D3              | D4              | D5              |
|--------|-----------------|-----------------|-----------------|-----------------|-----------------|
| S1     | 269.39105880208 | 1.45373353909   | 99.60345345757  | 26.64078166310  | 9.53768895688   |
| S2     | 9.29184687678   | 10.87477843707  | 144.52609291615 | 11.42013307790  | 153.17568199278 |
| S3     | 9.67458430167   | 2.61916509597   | 100.82422491687 | 3.21219108879   | 133.84933961242 |
| S4     | 270.57498480010 | 32.50253586     | 4.68420980965   | 1.56822696865   | 9.58927599      |
| S5     | 226.03319106758 | 8.66916198083   | 65.47681316968  | 9.06876525846   | 202.65015316426 |

##### Complete Model

$\begin{align*}
\min\quad & \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} \\
\text{s.t.}\quad
& \sum_{i \in I} x_{ij} = d_j,\quad \forall j \in J \\
& \sum_{j \in J} x_{ij} \leq s_i,\quad \forall i \in I \\
& x_{ij} \geq 0,\quad \forall i \in I,\, j \in J
\end{align*}$

With all parameters and indices as listed above.