##### Sets and Parameters

- Let $I = \{L1, L2, L3, L4, L5, L6, L7\}$ be the set of candidate facility locations.
- Let $J = \{A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12\}$ be the set of residential areas.
- Let $d_j$ be the demand of area $j \in J$:

\[
\begin{align*}
d_{A1} &= 25 \\
d_{A2} &= 35 \\
d_{A3} &= 40 \\
d_{A4} &= 30 \\
d_{A5} &= 50 \\
d_{A6} &= 45 \\
d_{A7} &= 20 \\
d_{A8} &= 55 \\
d_{A9} &= 60 \\
d_{A10} &= 30 \\
d_{A11} &= 42 \\
d_{A12} &= 38 \\
\end{align*}
\]

- Let $c_{ij}$ be the distance from location $i$ to area $j$:

\[
\begin{array}{c|cccccccccccc}
      & A1 & A2 & A3 & A4 & A5 & A6 & A7 & A8 & A9 & A10 & A11 & A12 \\
\hline
L1 & 2 & 3 & 4 & 8 & 9 & 10 & 13 & 14 & 15 & 12 & 11 & 10 \\
L2 & 3 & 2 & 3 & 7 & 8 & 9 & 12 & 13 & 14 & 11 & 10 & 9 \\
L3 & 8 & 7 & 5 & 2 & 3 & 4 & 8 & 9 & 11 & 7 & 6 & 7 \\
L4 & 9 & 8 & 6 & 3 & 2 & 3 & 7 & 8 & 10 & 6 & 5 & 6 \\
L5 & 13 & 12 & 10 & 8 & 7 & 6 & 2 & 3 & 4 & 5 & 6 & 7 \\
L6 & 14 & 13 & 11 & 9 & 8 & 7 & 3 & 2 & 3 & 4 & 5 & 6 \\
L7 & 11 & 10 & 8 & 7 & 6 & 5 & 6 & 5 & 4 & 2 & 3 & 2 \\
\end{array}
\]

- The required number of facilities to open: $p = 3$.

##### Decision Variables

- $y_i \in \{0,1\}$: 1 if facility at location $i \in I$ is opened, 0 otherwise.
- $x_{ij} \in \{0,1\}$: 1 if area $j \in J$ is assigned to facility $i \in I$, 0 otherwise.

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} d_j \cdot c_{ij} \cdot x_{ij}
\]

##### Constraints

1. **Assignment:** Each area is assigned to exactly one facility:
   \[
   \sum_{i \in I} x_{ij} = 1 \quad \forall j \in J
   \]

2. **Facility Opening:** Exactly $p$ facilities are opened:
   \[
   \sum_{i \in I} y_i = 3
   \]

3. **Assignment-to-Open-Facility Linking:** Areas can only be assigned to open facilities:
   \[
   x_{ij} \leq y_i \quad \forall i \in I,\, j \in J
   \]

4. **Binary Restrictions:**
   \[
   x_{ij} \in \{0,1\} \quad \forall i \in I,\, j \in J
   \]
   \[
   y_i \in \{0,1\} \quad \forall i \in I
   \]

##### Retrieved Information

- Areas $J$: A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12
- Demands $d_j$:
  - A1: 25
  - A2: 35
  - A3: 40
  - A4: 30
  - A5: 50
  - A6: 45
  - A7: 20
  - A8: 55
  - A9: 60
  - A10: 30
  - A11: 42
  - A12: 38
- Locations $I$: L1, L2, L3, L4, L5, L6, L7
- Distances $c_{ij}$: as in the table above
- Number of facilities to open $p$: 3