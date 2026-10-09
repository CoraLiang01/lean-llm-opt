##### Decision Variables

- $x_{ij} \geq 0$: quantity shipped from supplier $i \in I$ to branch $j \in J$ (continuous).
- $y_i \in \{0,1\}$: whether supplier $i$ is activated (binary).

##### Parameters

- Suppliers $I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}, \text{S5}\}$
- Branches $J = \{\text{C1}, \text{C2}, \text{C3}, \text{C4}, \text{C5}\}$

- Demands:
  - $d_{\text{C1}} = 143$
  - $d_{\text{C2}} = 6$
  - $d_{\text{C3}} = 10$
  - $d_{\text{C4}} = 25$
  - $d_{\text{C5}} = 3$

- Fixed costs:
  - $f_{\text{S1}} = 97.65$
  - $f_{\text{S2}} = 99.76$
  - $f_{\text{S3}} = 100.76$
  - $f_{\text{S4}} = 105.32$
  - $f_{\text{S5}} = 98.88$

- Transportation costs $c_{ij}$:

|        | C1      | C2      | C3      | C4      | C5      |
|--------|---------|---------|---------|---------|---------|
| S1     | 150.74  | 0.02    | 49.13   | 2080.15 | 426.4   |
| S2     | 233.05  | 97.73   | 49.84   | 1982.39 | 23.96   |
| S3     | 55.68   | 935.61  | 4.03    | 73.09   | 525.32  |
| S4     | 1483.82 | 1801.08 | 112.16  | 816.05  | 107.01  |
| S5     | 1119.47 | 884.31  | 0.08    | 1544.95 | 543.67  |

- $M = \sum_{j \in J} d_j = 143 + 6 + 10 + 25 + 3 = 187$

##### Objective Function

\[
\min \left( \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i \right)
\]

##### Constraints

1. **Demand satisfaction:**  
   For each branch $j \in J$,
   \[
   \sum_{i \in I} x_{ij} = d_j
   \]

2. **Supplier activation:**  
   For each supplier $i \in I$,
   \[
   \sum_{j \in J} x_{ij} \leq M y_i
   \]
   (A supplier can only ship goods if it is activated.)

3. **Variable domains:**  
   \[
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   \]
   \[
   y_i \in \{0,1\} \quad \forall i \in I
   \]

##### Full Model Data

- $I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}, \text{S5}\}$
- $J = \{\text{C1}, \text{C2}, \text{C3}, \text{C4}, \text{C5}\}$
- $d = \{ \text{C1}: 143,\, \text{C2}: 6,\, \text{C3}: 10,\, \text{C4}: 25,\, \text{C5}: 3 \}$
- $f = \{ \text{S1}: 97.65,\, \text{S2}: 99.76,\, \text{S3}: 100.76,\, \text{S4}: 105.32,\, \text{S5}: 98.88 \}$
- $c =$
  - S1: $\{ \text{C1}: 150.74,\, \text{C2}: 0.02,\, \text{C3}: 49.13,\, \text{C4}: 2080.15,\, \text{C5}: 426.4 \}$
  - S2: $\{ \text{C1}: 233.05,\, \text{C2}: 97.73,\, \text{C3}: 49.84,\, \text{C4}: 1982.39,\, \text{C5}: 23.96 \}$
  - S3: $\{ \text{C1}: 55.68,\, \text{C2}: 935.61,\, \text{C3}: 4.03,\, \text{C4}: 73.09,\, \text{C5}: 525.32 \}$
  - S4: $\{ \text{C1}: 1483.82,\, \text{C2}: 1801.08,\, \text{C3}: 112.16,\, \text{C4}: 816.05,\, \text{C5}: 107.01 \}$
  - S5: $\{ \text{C1}: 1119.47,\, \text{C2}: 884.31,\, \text{C3}: 0.08,\, \text{C4}: 1544.95,\, \text{C5}: 543.67 \}$
- $M = 187$