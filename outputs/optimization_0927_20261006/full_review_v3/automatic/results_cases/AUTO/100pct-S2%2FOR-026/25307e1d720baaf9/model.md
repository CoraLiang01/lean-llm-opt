##### Decision Variables

- $x_{ij} \geq 0$: quantity shipped from facility $i \in I$ to customer $j \in J$ (continuous).
- $y_i \in \{0,1\}$: whether facility $i$ is opened (binary).

##### Parameters

- $I = \{\text{F1}, \text{F2}, \ldots, \text{F15}\}$: set of candidate facilities.
- $J = \{\text{C1}, \text{C2}, \ldots, \text{C15}\}$: set of customers.

- Facility fixed opening costs $f_i$:
  - $f_{\text{F1}} = 11250$
  - $f_{\text{F2}} = 13480$
  - $f_{\text{F3}} = 14870$
  - $f_{\text{F4}} = 10290$
  - $f_{\text{F5}} = 16740$
  - $f_{\text{F6}} = 13960$
  - $f_{\text{F7}} = 12680$
  - $f_{\text{F8}} = 17890$
  - $f_{\text{F9}} = 10950$
  - $f_{\text{F10}} = 15320$
  - $f_{\text{F11}} = 11830$
  - $f_{\text{F12}} = 14110$
  - $f_{\text{F13}} = 15970$
  - $f_{\text{F14}} = 13140$
  - $f_{\text{F15}} = 10580$

- Facility capacities $u_i$:
  - $u_{\text{F1}} = 101$
  - $u_{\text{F2}} = 124$
  - $u_{\text{F3}} = 139$
  - $u_{\text{F4}} = 86$
  - $u_{\text{F5}} = 157$
  - $u_{\text{F6}} = 133$
  - $u_{\text{F7}} = 118$
  - $u_{\text{F8}} = 162$
  - $u_{\text{F9}} = 92$
  - $u_{\text{F10}} = 144$
  - $u_{\text{F11}} = 107$
  - $u_{\text{F12}} = 129$
  - $u_{\text{F13}} = 151$
  - $u_{\text{F14}} = 113$
  - $u_{\text{F15}} = 85$

- Customer demands $d_j$:
  - $d_{\text{C1}} = 83$
  - $d_{\text{C2}} = 76$
  - $d_{\text{C3}} = 91$
  - $d_{\text{C4}} = 68$
  - $d_{\text{C5}} = 104$
  - $d_{\text{C6}} = 97$
  - $d_{\text{C7}} = 88$
  - $d_{\text{C8}} = 73$
  - $d_{\text{C9}} = 109$
  - $d_{\text{C10}} = 95$
  - $d_{\text{C11}} = 82$
  - $d_{\text{C12}} = 67$
  - $d_{\text{C13}} = 113$
  - $d_{\text{C14}} = 79$
  - $d_{\text{C15}} = 92$

- Per-unit transportation costs $c_{ij}$ (for $i \in I$, $j \in J$):

|        | C1  | C2  | C3  | C4  | C5  | C6  | C7  | C8  | C9  | C10 | C11 | C12 | C13 | C14 | C15 |
|--------|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| F1     | 7.8 | 7.6 | 6.7 | 7.9 | 8.1 | 8.3 | 7.3 | 8.2 | 8.1 | 8.2 | 7.3 | 7.7 | 6.7 | 7.1 | 7.9 |
| F2     | 5.3 | 6   | 5   | 6.4 | 5.9 | 6.2 | 5.6 | 6.1 | 6.3 | 6.1 | 5   | 5.6 | 5.3 | 4.9 | 6.3 |
| F3     | 7.2 | 8.1 | 7.4 | 8.8 | 8.5 | 8.7 | 7.7 | 8.7 | 8.9 | 8.5 | 7.2 | 7.7 | 7.1 | 7.6 | 8.4 |
| F4     | 7   | 7.1 | 6.5 | 7.9 | 7.4 | 7.7 | 6.7 | 7.9 | 7.8 | 7.3 | 6.8 | 7   | 6.5 | 6.7 | 7.6 |
| F5     | 3.5 | 3.8 | 2.9 | 4.3 | 3.6 | 3.9 | 3.2 | 4.3 | 4.5 | 4   | 3.2 | 4   | 2.9 | 3.4 | 3.9 |
| F6     | 8.2 | 8.6 | 7.9 | 9.5 | 8.5 | 9.3 | 8.5 | 9.4 | 9   | 9.2 | 8.1 | 8.7 | 7.9 | 8.5 | 9   |
| F7     | 6.9 | 7.6 | 6.8 | 8.4 | 8   | 8   | 7.6 | 8   | 8.1 | 7.8 | 6.9 | 7.1 | 7   | 6.9 | 7.5 |
| F8     | 6.9 | 7.8 | 7.1 | 8.7 | 8.6 | 8.2 | 7.2 | 7.9 | 8.4 | 7.9 | 7   | 7.4 | 6.8 | 7.3 | 8   |
| F9     | 3.5 | 3.8 | 2.8 | 4.4 | 4.2 | 4.8 | 3.8 | 5   | 4.5 | 4.1 | 3.2 | 3.7 | 3.7 | 3.2 | 4.5 |
| F10    | 5.2 | 6.1 | 5.1 | 6.3 | 6.1 | 6   | 5.6 | 6.5 | 6.2 | 5.9 | 5.3 | 6.1 | 5.1 | 5.2 | 6.2 |
| F11    | 5.2 | 5.5 | 4.5 | 6.2 | 5.7 | 6.1 | 5.1 | 5.8 | 5.7 | 6.2 | 5.2 | 5.2 | 4.5 | 5.1 | 5.4 |
| F12    | 7.8 | 8.7 | 7.6 | 9   | 8.6 | 9   | 8.5 | 9.3 | 9.3 | 8.4 | 7.9 | 8.2 | 7.4 | 7.6 | 8.7 |
| F13    | 6.7 | 6.6 | 6.1 | 7.3 | 7.1 | 7.5 | 6.7 | 8   | 7.6 | 7.2 | 6.3 | 6.9 | 6.2 | 6   | 7.2 |
| F14    | 7.5 | 8.6 | 7.6 | 8.2 | 8   | 7.9 | 7.5 | 8.7 | 8.8 | 8.1 | 7.2 | 7.3 | 7   | 7   | 8   |
| F15    | 5.1 | 5.8 | 4.6 | 5.9 | 6.5 | 5.9 | 5.2 | 7   | 7.1 | 5.9 | 5.1 | 5.8 | 5.4 | 4.9 | 6   |

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
\]

##### Constraints

1. **Demand satisfaction:**  
   For each customer $j \in J$,
   \[
   \sum_{i \in I} x_{ij} = d_j
   \]

2. **Facility capacity:**  
   For each facility $i \in I$,
   \[
   \sum_{j \in J} x_{ij} \leq u_i y_i
   \]

3. **Variable domains:**  
   \[
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   \]
   \[
   y_i \in \{0,1\} \quad \forall i \in I
   \]

##### All Parameters (as retrieved)

- Facilities: $I = \{\text{F1}, \text{F2}, \ldots, \text{F15}\}$
- Customers: $J = \{\text{C1}, \text{C2}, \ldots, \text{C15}\}$
- Facility fixed opening costs $f_i$ and capacities $u_i$ as listed above.
- Customer demands $d_j$ as listed above.
- Per-unit transportation costs $c_{ij}$ as in the table above.

This is the complete mathematical model for the capacitated facility location problem as described.