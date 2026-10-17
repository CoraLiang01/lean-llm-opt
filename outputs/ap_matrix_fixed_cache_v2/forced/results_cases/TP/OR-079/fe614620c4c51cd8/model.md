##### Sets

- $I = \{A1, A2, \ldots, A15\}$: potential factory sites  
- $J = \{B1, B2, \ldots, B8\}$: distribution centers

##### Parameters

- $f_i$: fixed cost to open facility $i$  
- $K_i$: capacity of facility $i$  
- $d_j$: demand at distribution center $j$  
- $c_{ij}$: shipping cost per unit from facility $i$ to center $j$

From the retrieved data (source order):

Facility fixed costs and capacities:
- $f_{A1} = 0$, $K_{A1} = 30$
- $f_{A2} = 175$, $K_{A2} = 10$
- $f_{A3} = 300$, $K_{A3} = 20$
- $f_{A4} = 375$, $K_{A4} = 30$
- $f_{A5} = 500$, $K_{A5} = 40$
- $f_{A6} = 200$, $K_{A6} = 20$
- $f_{A7} = 260$, $K_{A7} = 25$
- $f_{A8} = 220$, $K_{A8} = 30$
- $f_{A9} = 320$, $K_{A9} = 35$
- $f_{A10} = 280$, $K_{A10} = 20$
- $f_{A11} = 350$, $K_{A11} = 40$
- $f_{A12} = 420$, $K_{A12} = 25$
- $f_{A13} = 470$, $K_{A13} = 30$
- $f_{A14} = 520$, $K_{A14} = 50$
- $f_{A15} = 560$, $K_{A15} = 45$

Distribution center demands:
- $d_{B1} = 30$
- $d_{B2} = 25$
- $d_{B3} = 20$
- $d_{B4} = 35$
- $d_{B5} = 25$
- $d_{B6} = 30$
- $d_{B7} = 25$
- $d_{B8} = 30$

Shipping costs $c_{ij}$ (rows: $i$ in $I$, columns: $j$ in $J$):

|      | B1 | B2 | B3 | B4 | B5 | B6 | B7 | B8 |
|------|----|----|----|----|----|----|----|----|
| A1   | 8  | 4  | 3  | 6  | 7  | 5  | 9  | 8  |
| A2   | 5  | 2  | 3  | 5  | 6  | 4  | 7  | 6  |
| A3   | 4  | 3  | 4  | 6  | 5  | 5  | 6  | 7  |
| A4   | 9  | 7  | 5  | 8  | 9  | 6  | 10 | 7  |
| A5   | 10 | 4  | 2  | 6  | 8  | 5  | 7  | 3  |
| A6   | 6  | 5  | 4  | 5  | 7  | 6  | 8  | 5  |
| A7   | 7  | 6  | 5  | 4  | 6  | 7  | 9  | 6  |
| A8   | 5  | 4  | 6  | 3  | 5  | 6  | 7  | 6  |
| A9   | 8  | 7  | 6  | 7  | 9  | 8  | 10 | 7  |
| A10  | 6  | 5  | 7  | 4  | 6  | 5  | 7  | 5  |
| A11  | 9  | 6  | 4  | 6  | 8  | 7  | 9  | 6  |
| A12  | 7  | 5  | 6  | 5  | 6  | 5  | 8  | 5  |
| A13  | 8  | 6  | 5  | 6  | 7  | 6  | 8  | 7  |
| A14  | 9  | 5  | 3  | 5  | 7  | 4  | 6  | 4  |
| A15  | 10 | 6  | 4  | 5  | 8  | 5  | 7  | 5  |

##### Decision Variables

- $y_i \in \{0,1\}$: 1 if facility $i$ is built, 0 otherwise
- $x_{ij} \geq 0$: amount shipped from facility $i$ to center $j$ (continuous)

##### Objective

Minimize total cost:
$$
\min \sum_{i \in I} f_i y_i + \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
$$

##### Constraints

1. **Demand satisfaction:**  
   $$
   \sum_{i \in I} x_{ij} = d_j \qquad \forall j \in J
   $$

2. **Facility capacity (only if open):**  
   $$
   \sum_{j \in J} x_{ij} \leq K_i y_i \qquad \forall i \in I
   $$

3. **Variable domains:**  
   $$
   y_i \in \{0,1\} \qquad \forall i \in I
   $$
   $$
   x_{ij} \geq 0 \qquad \forall i \in I,\, j \in J
   $$

##### All identifiers and coefficients are preserved as retrieved.

###### Retrieved Information

- Facilities: $A1$–$A15$ with fixed costs and capacities as above
- Distribution centers: $B1$–$B8$ with demands as above
- Shipping costs: $c_{ij}$ as in the table above (source order)