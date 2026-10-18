##### Sets and Indices

Let $I$ be the set of all products classified as ‘FAUX’:
\[
I = \{
\text{FAUX FUR JEWEL SWEATER},
\text{FAUX LEATHER BOMBER JACKET},
\text{FAUX LEATHER BOXY FIT JACKET},
\text{FAUX LEATHER JACKET},
\text{FAUX LEATHER OVERSIZED JACKET LIMITED EDITION},
\text{FAUX LEATHER PUFFER JACKET},
\text{FAUX SHEARLING LINED SUEDE BOOTS},
\text{FAUX SHEARLING PLAID JACKET},
\text{FAUX SUEDE BOMBER JACKET},
\text{FAUX SUEDE JACKET},
\text{FAUX SUEDE OVERSHIRT},
\text{FAUX SUEDE PATCH JACKET}
\}
\]

##### Parameters

For each $i \in I$:

- $r_i$: Revenue per unit of product $i$
- $s_i$: Initial inventory of product $i$
- $d_i$: Demand for product $i$

The parameter values are:

| Product Name                                         | $r_i$   | $s_i$  | $d_i$  |
|------------------------------------------------------|---------|--------|--------|
| FAUX FUR JEWEL SWEATER                               | 35.9    | 20970  | 3025   |
| FAUX LEATHER BOMBER JACKET                           | 69.9    | 71970  | 9585   |
| FAUX LEATHER BOXY FIT JACKET                         | 99.9    | 32730  | 4486   |
| FAUX LEATHER JACKET                                  | 99.9    | 71130  | 10322  |
| FAUX LEATHER OVERSIZED JACKET LIMITED EDITION        | 159.0   | 34910  | 4868   |
| FAUX LEATHER PUFFER JACKET                           | 69.99   | 64010  | 8482   |
| FAUX SHEARLING LINED SUEDE BOOTS                     | 99.9    | 20760  | 2607   |
| FAUX SHEARLING PLAID JACKET                          | 89.9    | 12490  | 1784   |
| FAUX SUEDE BOMBER JACKET                             | 69.9    | 50300  | 6626   |
| FAUX SUEDE JACKET                                    | 89.9    | 24570  | 3256   |
| FAUX SUEDE OVERSHIRT                                 | 69.9    | 24430  | 2955   |
| FAUX SUEDE PATCH JACKET                              | 89.9    | 7070   | 910    |

##### Decision Variables

For each $i \in I$:

- $x_i \geq 0$: Number of units of product $i$ to fulfill (continuous or integer, as appropriate).

##### Objective Function

\[
\max \sum_{i \in I} r_i x_i
\]

##### Constraints

1. Inventory and demand limits:
   \[
   0 \leq x_i \leq \min\{s_i, d_i\}, \quad \forall i \in I
   \]

##### Complete Model

\[
\begin{align*}
\max \quad & \sum_{i \in I} r_i x_i \\
\text{s.t.} \quad & 0 \leq x_i \leq \min\{s_i, d_i\}, \quad \forall i \in I \\
\end{align*}
\]

Where the parameters are as listed above.