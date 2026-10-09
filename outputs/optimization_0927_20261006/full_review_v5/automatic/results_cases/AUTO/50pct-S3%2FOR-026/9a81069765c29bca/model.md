##### Sets

- $I = \{F1, F2, \ldots, F15\}$: set of candidate plants.
- $J = \{C1, C2, \ldots, C15\}$: set of customers.

##### Parameters

- $f_i$: fixed opening cost for plant $i \in I$.
- $K_i$: capacity of plant $i \in I$.
- $c_{ij}$: per-unit transportation cost from plant $i$ to customer $j$.
- $d_j$: demand of customer $j \in J$.

###### Plant data (from cost.csv):

| Plant | $f_i$ (fixed opening cost) | $K_i$ (capacity) | $c_{i,C1}$ | $c_{i,C2}$ | $c_{i,C3}$ | $c_{i,C4}$ | $c_{i,C5}$ | $c_{i,C6}$ | $c_{i,C7}$ | $c_{i,C8}$ | $c_{i,C9}$ | $c_{i,C10}$ | $c_{i,C11}$ | $c_{i,C12}$ | $c_{i,C13}$ | $c_{i,C14}$ | $c_{i,C15}$ |
|-------|----------------------------|------------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|-------------|-------------|-------------|-------------|-------------|-------------|
| F1    | 11250                      | 101              | 7.8        | 7.6        | 6.7        | 7.9        | 8.1        | 8.3        | 7.3        | 8.2        | 8.1        | 8.2         | 7.3         | 7.7         | 6.7         | 7.1         | 7.9         |
| F2    | 13480                      | 124              | 5.3        | 6          | 5          | 6.4        | 5.9        | 6.2        | 5.6        | 6.1        | 6.3        | 6.1         | 5           | 5.6         | 5.3         | 4.9         | 6.3         |
| F3    | 14870                      | 139              | 7.2        | 8.1        | 7.4        | 8.8        | 8.5        | 8.7        | 7.7        | 8.7        | 8.9        | 8.5         | 7.2         | 7.7         | 7.1         | 7.6         | 8.4         |
| F4    | 10290                      | 86               | 7          | 7.1        | 6.5        | 7.9        | 7.4        | 7.7        | 6.7        | 7.9        | 7.8        | 7.3         | 6.8         | 7           | 6.5         | 6.7         | 7.6         |
| F5    | 16740                      | 157              | 3.5        | 3.8        | 2.9        | 4.3        | 3.6        | 3.9        | 3.2        | 4.3        | 4.5        | 4           | 3.2         | 4           | 2.9         | 3.4         | 3.9         |
| F6    | 13960                      | 133              | 8.2        | 8.6        | 7.9        | 9.5        | 8.5        | 9.3        | 8.5        | 9.4        | 9          | 9.2         | 8.1         | 8.7         | 7.9         | 8.5         | 9           |
| F7    | 12680                      | 118              | 6.9        | 7.6        | 6.8        | 8.4        | 8          | 8          | 7.6        | 8          | 8.1        | 7.8         | 6.9         | 7.1         | 7           | 6.9         | 7.5         |
| F8    | 17890                      | 162              | 6.9        | 7.8        | 7.1        | 8.7        | 8.6        | 8.2        | 7.2        | 7.9        | 8.4        | 7.9         | 7           | 7.4         | 6.8         | 7.3         | 8           |
| F9    | 10950                      | 92               | 3.5        | 3.8        | 2.8        | 4.4        | 4.2        | 4.8        | 3.8        | 5          | 4.5        | 4.1         | 3.2         | 3.7         | 3.7         | 3.2         | 4.5         |
| F10   | 15320                      | 144              | 5.2        | 6.1        | 5.1        | 6.3        | 6.1        | 6          | 5.6        | 6.5        | 6.2        | 5.9         | 5.3         | 6.1         | 5.1         | 5.2         | 6.2         |
| F11   | 11830                      | 107              | 5.2        | 5.5        | 4.5        | 6.2        | 5.7        | 6.1        | 5.1        | 5.8        | 5.7        | 6.2         | 5.2         | 5.2         | 4.5         | 5.1         | 5.4         |
| F12   | 14110                      | 129              | 7.8        | 8.7        | 7.6        | 9          | 8.6        | 9          | 8.5        | 9.3        | 9.3        | 8.4         | 7.9         | 8.2         | 7.4         | 7.6         | 8.7         |
| F13   | 15970                      | 151              | 6.7        | 6.6        | 6.1        | 7.3        | 7.1        | 7.5        | 6.7        | 8          | 7.6        | 7.2         | 6.3         | 6.9         | 6.2         | 6           | 7.2         |
| F14   | 13140                      | 113              | 7.5        | 8.6        | 7.6        | 8.2        | 8          | 7.9        | 7.5        | 8.7        | 8.8        | 8.1         | 7.2         | 7.3         | 7           | 7           | 8           |
| F15   | 10580                      | 85               | 5.1        | 5.8        | 4.6        | 5.9        | 6.5        | 5.9        | 5.2        | 7          | 7.1        | 5.9         | 5.1         | 5.8         | 5.4         | 4.9         | 6           |

###### Customer demand (from demand.csv):

| Customer | $d_j$ (demand) |
|----------|----------------|
| C1       | 83             |
| C2       | 76             |
| C3       | 91             |
| C4       | 68             |
| C5       | 104            |
| C6       | 97             |
| C7       | 88             |
| C8       | 73             |
| C9       | 109            |
| C10      | 95             |
| C11      | 82             |
| C12      | 67             |
| C13      | 113            |
| C14      | 79             |
| C15      | 92             |

##### Decision Variables

- $y_i \in \{0,1\}$: 1 if plant $i$ is built, 0 otherwise.
- $x_{ij} \geq 0$: quantity shipped from plant $i$ to customer $j$.

##### Objective Function

\[
\min \sum_{i \in I} f_i y_i + \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

##### Constraints

1. **Demand satisfaction:**  
   For each customer $j \in J$,
   \[
   \sum_{i \in I} x_{ij} = d_j
   \]

2. **Plant capacity:**  
   For each plant $i \in I$,
   \[
   \sum_{j \in J} x_{ij} \leq K_i y_i
   \]

3. **Variable domains:**  
   \[
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   \]
   \[
   y_i \in \{0,1\} \quad \forall i \in I
   \]

##### All required parameters:

- $I = \{F1, F2, \ldots, F15\}$
- $J = \{C1, C2, \ldots, C15\}$
- $f_i$, $K_i$, $c_{ij}$ as in the table above
- $d_j$ as in the table above

This is the complete mathematical model for the described facility location problem with capacity constraints.