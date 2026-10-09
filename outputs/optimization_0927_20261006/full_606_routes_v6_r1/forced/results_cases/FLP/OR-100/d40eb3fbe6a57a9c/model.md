##### Sets

- $C = \{C1, C2, \ldots, C111\}$: set of component types.
- $W = \{\text{Casting}, \text{Milling}, \text{Finishing}, \text{Assembly}, \text{QA \& Packaging}\}$: set of workshops.

##### Parameters

- $p_j$: unit price of component $j \in C$.
- $a_{wj}$: unit processing time (hours) required for component $j$ in workshop $w$.
- $T_w$: total available working hours in workshop $w$.

**Unit processing times $a_{wj}$ (from processing_time_unit.csv):**

- For each $w \in W$, $j \in C$, $a_{wj}$ is as follows (partial, see full CSV for all values):

| Workshop         | C1   | C2   | C3   | ... | C111 |
|------------------|------|------|------|-----|------|
| Casting          | 0.74 | 0.77 | 1.41 | ... | 3.81 |
| Milling          | 0.60 | 3.38 | 0.00 | ... | 2.67 |
| Finishing        | 0.00 | 4.15 | 0.00 | ... | 4.07 |
| Assembly         | 4.84 | 0.00 | 3.80 | ... | 4.02 |
| QA & Packaging   | 0.92 | 0.00 | 1.08 | ... | 3.21 |

**Unit prices $p_j$ (from unit_price.csv):**

- $p_{C1} = 193$, $p_{C2} = 64$, $p_{C3} = 103$, ..., $p_{C111} = 142$

**Total available working hours $T_w$ (from total_working_hours.csv):**

- $T_{\text{Casting}} = 7650$
- $T_{\text{Milling}} = 6320$
- $T_{\text{Finishing}} = 5538$
- $T_{\text{Assembly}} = 5957$
- $T_{\text{QA \& Packaging}} = 6988$

##### Decision Variables

- $x_j \geq 0$: production quantity of component $j \in C$ (continuous).

##### Objective Function

\[
\max \sum_{j \in C} p_j x_j
\]

##### Constraints

1. **Workshop capacity constraints:**

   For each workshop $w \in W$,
   \[
   \sum_{j \in C} a_{wj} x_j \leq T_w
   \]

2. **Nonnegativity:**

   \[
   x_j \geq 0, \quad \forall j \in C
   \]

##### Full Parameter Listing

- $C = \{C1, C2, ..., C111\}$
- $W = \{\text{Casting}, \text{Milling}, \text{Finishing}, \text{Assembly}, \text{QA \& Packaging}\}$
- $p_j$ for all $j \in C$ as listed above.
- $a_{wj}$ for all $w \in W$, $j \in C$ as in processing_time_unit.csv.
- $T_w$ for all $w \in W$ as listed above.

##### Mathematical Model

\[
\begin{align*}
\max \quad & \sum_{j \in C} p_j x_j \\
\text{s.t.} \quad & \sum_{j \in C} a_{wj} x_j \leq T_w, \quad \forall w \in W \\
& x_j \geq 0, \quad \forall j \in C
\end{align*}
\]

Where all parameters are as retrieved from the CSV files above.