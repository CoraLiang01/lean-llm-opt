##### Sets and Indices

Let $I = \{1,2,\ldots,100\}$ index the products, with product $i$ corresponding to $P_i$.

Let $K = \{\text{R1}, \text{R2}, \text{R3}\}$ index the resources.

##### Parameters (from CSV, source order preserved)

For each product $i$:

- $profit\_per\_unit_i$ = profit per unit of $P_i$
- $r1\_per\_unit_i$ = units of R1 consumed per unit of $P_i$
- $r2\_per\_unit_i$ = units of R2 consumed per unit of $P_i$
- $r3\_per\_unit_i$ = units of R3 consumed per unit of $P_i$
- $upper\_demand\_units_i$ = maximum market demand (units) for $P_i$
- $batch\_size\_units_i = 10$ (for all $i$)

Resource capacities:

- $capacity_{\text{R1}} = 27380.54$
- $capacity_{\text{R2}} = 22245.11$
- $capacity_{\text{R3}} = 15147.73$

##### Decision Variables

For each product $i$:

- $x_i \in \mathbb{Z}_+$: number of batches of product $i$ to produce (integer, $x_i \geq 0$)

##### Objective Function

\[
\max \sum_{i=1}^{100} \left(10 \cdot x_i \cdot profit\_per\_unit_i\right)
\]

##### Constraints

For each resource $k$:

- R1:
  \[
  \sum_{i=1}^{100} 10 \cdot x_i \cdot r1\_per\_unit_i \leq 27380.54
  \]
- R2:
  \[
  \sum_{i=1}^{100} 10 \cdot x_i \cdot r2\_per\_unit_i \leq 22245.11
  \]
- R3:
  \[
  \sum_{i=1}^{100} 10 \cdot x_i \cdot r3\_per\_unit_i \leq 15147.73
  \]

For each product $i$ (demand upper bound):

\[
10 \cdot x_i \leq upper\_demand\_units_i
\]

For all $i$:

\[
x_i \in \mathbb{Z}_+, \quad x_i \geq 0
\]

##### Data (source order, all identifiers and coefficients)

For $i=1$ to $100$ (products $P1$ to $P100$):

| $i$ | product | $profit\_per\_unit_i$ | $r1\_per\_unit_i$ | $r2\_per\_unit_i$ | $r3\_per\_unit_i$ | $upper\_demand\_units_i$ | $batch\_size\_units_i$ |
|----|---------|----------------------|------------------|------------------|------------------|-------------------------|----------------------|
| 1 | P1 | 6.7 | 2.37 | 0.61 | 2.03 | 317 | 10 |
| 2 | P2 | 10.96 | 4.79 | 2.73 | 0.53 | 106 | 10 |
| 3 | P3 | 9.4 | 3.87 | 1.6 | 0.74 | 386 | 10 |
| 4 | P4 | 11.13 | 3.31 | 2.28 | 2.73 | 441 | 10 |
| 5 | P5 | 10.29 | 1.46 | 3.68 | 1.94 | 63 | 10 |
| 6 | P6 | 8.06 | 1.46 | 1.37 | 0.32 | 221 | 10 |
| 7 | P7 | 6.94 | 1.04 | 1.94 | 0.57 | 441 | 10 |
| 8 | P8 | 11.44 | 4.44 | 3.14 | 2.09 | 489 | 10 |
| 9 | P9 | 9.13 | 3.32 | 1.3 | 0.31 | 277 | 10 |
| 10 | P10 | 7.83 | 3.77 | 0.77 | 0.73 | 121 | 10 |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 100 | P100 | 8.84 | 1.25 | 3.23 | 0.53 | 576 | 10 |

Resource capacities:

- $capacity_{\text{R1}} = 27380.54$
- $capacity_{\text{R2}} = 22245.11$
- $capacity_{\text{R3}} = 15147.73$

##### Complete Model

\[
\begin{align*}
\max\ & \sum_{i=1}^{100} 10\, x_i\, profit\_per\_unit_i \\
\text{s.t.}\quad
& \sum_{i=1}^{100} 10\, x_i\, r1\_per\_unit_i \leq 27380.54 \\
& \sum_{i=1}^{100} 10\, x_i\, r2\_per\_unit_i \leq 22245.11 \\
& \sum_{i=1}^{100} 10\, x_i\, r3\_per\_unit_i \leq 15147.73 \\
& 10\, x_i \leq upper\_demand\_units_i \quad \forall i=1,\ldots,100 \\
& x_i \in \mathbb{Z}_+, \quad x_i \geq 0 \quad \forall i=1,\ldots,100
\end{align*}
\]

where all coefficients and bounds are as listed above, in the retrieved data.