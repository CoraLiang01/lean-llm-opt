##### Decision Variables

Let $x_j \geq 0$ denote the production quantity of component $j$, for $j \in J = \{C1, C2, \ldots, C111\}$.

##### Parameters

- $P_j$: unit price of component $j$ (from unit_price.csv)
- $a_{ij}$: unit processing time required for component $j$ in workshop $i$ (from processing_time_unit.csv), for $i \in I = \{\text{Casting}, \text{Milling}, \text{Finishing}, \text{Assembly}, \text{QA \& Packaging}\}$
- $T_i$: total available working hours in workshop $i$ (from total_working_hours.csv)

###### Sets

- $I = \{\text{Casting}, \text{Milling}, \text{Finishing}, \text{Assembly}, \text{QA \& Packaging}\}$
- $J = \{C1, C2, \ldots, C111\}$

##### Objective Function

\[
\max \sum_{j \in J} P_j x_j
\]

##### Constraints

For each workshop $i \in I$:
\[
\sum_{j \in J} a_{ij} x_j \leq T_i
\]

For each component $j \in J$:
\[
x_j \geq 0
\]

##### Data

- $P_j$ (unit price for each component $j$):

\[
\begin{aligned}
& P_{C1} = 193,\quad P_{C2} = 64,\quad P_{C3} = 103,\quad P_{C4} = 210,\quad P_{C5} = 85, \\
& P_{C6} = 126,\quad P_{C7} = 226,\quad P_{C8} = 94,\quad P_{C9} = 73,\quad P_{C10} = 120, \\
& \ldots \\
& P_{C111} = 142
\end{aligned}
\]
(Full vector as in unit_price.csv.)

- $a_{ij}$ (unit processing time for each workshop $i$ and component $j$):

For example, for $i = \text{Casting}$:
\[
(a_{\text{Casting},C1}, a_{\text{Casting},C2}, \ldots, a_{\text{Casting},C111}) = (0.74, 0.77, 1.41, 2.11, 2.19, 1.31, 3.75, 3.47, 3.8, 3.81, \ldots, 3.81)
\]
Similarly for Milling, Finishing, Assembly, QA & Packaging (see processing_time_unit.csv for all values).

- $T_i$ (total available working hours for each workshop $i$):

\[
\begin{aligned}
& T_{\text{Casting}} = 7650 \\
& T_{\text{Milling}} = 6320 \\
& T_{\text{Finishing}} = 5538 \\
& T_{\text{Assembly}} = 5957 \\
& T_{\text{QA \& Packaging}} = 6988
\end{aligned}
\]

##### Complete Mathematical Model

\[
\begin{aligned}
\max \quad & \sum_{j \in J} P_j x_j \\
\text{s.t.} \quad & \sum_{j \in J} a_{ij} x_j \leq T_i, \quad \forall i \in I \\
& x_j \geq 0, \quad \forall j \in J
\end{aligned}
\]

Where all $P_j$, $a_{ij}$, and $T_i$ are as listed above and in the retrieved CSV data.

###### Retrieved Information

- Components: $J = \{C1, C2, \ldots, C111\}$
- Workshops: $I = \{\text{Casting}, \text{Milling}, \text{Finishing}, \text{Assembly}, \text{QA \& Packaging}\}$
- $P_j$: as in unit_price.csv
- $a_{ij}$: as in processing_time_unit.csv (rows: workshops, columns: components)
- $T_i$: as in total_working_hours.csv

All vectors and matrices are to be used as retrieved, without abbreviation.