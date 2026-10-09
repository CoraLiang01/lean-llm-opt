##### Decision Variables

- $x_{ij} \geq 0$: quantity shipped from plant $i \in I$ to customer $j \in J$ (continuous).
- $y_i \in \{0,1\}$: whether plant $i$ is built (opened).

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
\]

where:
- $c_{ij}$: per-unit transportation cost from plant $i$ to customer $j$
- $f_i$: fixed opening cost for plant $i$

##### Constraints

1. **Demand satisfaction:**  
   \[
   \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J
   \]
   Each customer's demand must be fully met.

2. **Plant capacity (conditional on opening):**  
   \[
   \sum_{j \in J} x_{ij} \leq \text{cap}_i \cdot y_i, \quad \forall i \in I
   \]
   No plant can supply more than its capacity, and only if it is opened.

3. **Variable domains:**  
   \[
   x_{ij} \geq 0 \text{ continuous}, \quad y_i \in \{0,1\}
   \]

---

#### Sets

- $I = \{\text{F1}, \text{F2}, \ldots, \text{F15}\}$ (plants)
- $J = \{\text{C1}, \text{C2}, \ldots, \text{C15}\}$ (customers)

#### Parameters (retrieved from CSV)

- **Fixed opening costs $f_i$ and capacities $\text{cap}_i$:**

| Plant | $f_i$ (fixed opening cost) | $\text{cap}_i$ (capacity) |
|-------|---------------------------|---------------------------|
| F1    | 11250                     | 101                       |
| F2    | 13480                     | 124                       |
| F3    | 14870                     | 139                       |
| F4    | 10290                     | 86                        |
| F5    | 16740                     | 157                       |
| F6    | 13960                     | 133                       |
| F7    | 12680                     | 118                       |
| F8    | 17890                     | 162                       |
| F9    | 10950                     | 92                        |
| F10   | 15320                     | 144                       |
| F11   | 11830                     | 107                       |
| F12   | 14110                     | 129                       |
| F13   | 15970                     | 151                       |
| F14   | 13140                     | 113                       |
| F15   | 10580                     | 85                        |

- **Customer demands $d_j$:**

| Customer | $d_j$ (demand) |
|----------|---------------|
| C1       | 83            |
| C2       | 76            |
| C3       | 91            |
| C4       | 68            |
| C5       | 104           |
| C6       | 97            |
| C7       | 88            |
| C8       | 73            |
| C9       | 109           |
| C10      | 95            |
| C11      | 82            |
| C12      | 67            |
| C13      | 113           |
| C14      | 79            |
| C15      | 92            |

- **Transportation costs $c_{ij}$ (plant $i$ to customer $j$):**

| Plant | C1  | C2  | C3  | C4  | C5  | C6  | C7  | C8  | C9  | C10 | C11 | C12 | C13 | C14 | C15 |
|-------|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| F1    | 7.8 | 7.6 | 6.7 | 7.9 | 8.1 | 8.3 | 7.3 | 8.2 | 8.1 | 8.2 | 7.3 | 7.7 | 6.7 | 7.1 | 7.9 |
| F2    | 5.3 | 6   | 5   | 6.4 | 5.9 | 6.2 | 5.6 | 6.1 | 6.3 | 6.1 | 5   | 5.6 | 5.3 | 4.9 | 6.3 |
| F3    | 7.2 | 8.1 | 7.4 | 8.8 | 8.5 | 8.7 | 7.7 | 8.7 | 8.9 | 8.5 | 7.2 | 7.7 | 7.1 | 7.6 | 8.4 |
| F4    | 7   | 7.1 | 6.5 | 7.9 | 7.4 | 7.7 | 6.7 | 7.9 | 7.8 | 7.3 | 6.8 | 7   | 6.5 | 6.7 | 7.6 |
| F5    | 3.5 | 3.8 | 2.9 | 4.3 | 3.6 | 3.9 | 3.2 | 4.3 | 4.5 | 4   | 3.2 | 4   | 2.9 | 3.4 | 3.9 |
| F6    | 8.2 | 8.6 | 7.9 | 9.5 | 8.5 | 9.3 | 8.5 | 9.4 | 9   | 9.2 | 8.1 | 8.7 | 7.9 | 8.5 | 9   |
| F7    | 6.9 | 7.6 | 6.8 | 8.4 | 8   | 8   | 7.6 | 8   | 8.1 | 7.8 | 6.9 | 7.1 | 7   | 6.9 | 7.5 |
| F8    | 6.9 | 7.8 | 7.1 | 8.7 | 8.6 | 8.2 | 7.2 | 7.9 | 8.4 | 7.9 | 7   | 7.4 | 6.8 | 7.3 | 8   |
| F9    | 3.5 | 3.8 | 2.8 | 4.4 | 4.2 | 4.8 | 3.8 | 5   | 4.5 | 4.1 | 3.2 | 3.7 | 3.7 | 3.2 | 4.5 |
| F10   | 5.2 | 6.1 | 5.1 | 6.3 | 6.1 | 6   | 5.6 | 6.5 | 6.2 | 5.9 | 5.3 | 6.1 | 5.1 | 5.2 | 6.2 |
| F11   | 5.2 | 5.5 | 4.5 | 6.2 | 5.7 | 6.1 | 5.1 | 5.8 | 5.7 | 6.2 | 5.2 | 5.2 | 4.5 | 5.1 | 5.4 |
| F12   | 7.8 | 8.7 | 7.6 | 9   | 8.6 | 9   | 8.5 | 9.3 | 9.3 | 8.4 | 7.9 | 8.2 | 7.4 | 7.6 | 8.7 |
| F13   | 6.7 | 6.6 | 6.1 | 7.3 | 7.1 | 7.5 | 6.7 | 8   | 7.6 | 7.2 | 6.3 | 6.9 | 6.2 | 6   | 7.2 |
| F14   | 7.5 | 8.6 | 7.6 | 8.2 | 8   | 7.9 | 7.5 | 8.7 | 8.8 | 8.1 | 7.2 | 7.3 | 7   | 7   | 8   |
| F15   | 5.1 | 5.8 | 4.6 | 5.9 | 6.5 | 5.9 | 5.2 | 7   | 7.1 | 5.9 | 5.1 | 5.8 | 5.4 | 4.9 | 6   |

---

**Summary of Model:**

- Decision variables: $x_{ij}$ (continuous, shipment), $y_i$ (binary, plant open)
- Objective: Minimize total cost (fixed opening + transportation)
- Constraints: Demand satisfaction, plant capacity (if open), variable domains
- All parameters and indices as above, directly from the CSV data.