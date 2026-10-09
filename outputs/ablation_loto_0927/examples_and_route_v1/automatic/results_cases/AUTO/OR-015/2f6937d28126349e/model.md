Let $x_{\text{Aalopuri}}$ be the number of units of Aalopuri to fulfill.

Objective:
$$
\max\ 20\, x_{\text{Aalopuri}}
$$

Subject to:
$$
0 \leq x_{\text{Aalopuri}} \leq 1483 \\
0 \leq x_{\text{Aalopuri}} \leq 10440 \\
x_{\text{Aalopuri}} \in \mathbb{Z}_{\geq 0}
$$

Where:
- $x_{\text{Aalopuri}}$ = number of Aalopuri units fulfilled (integer, nonnegative)
- 20 = Revenue per unit (from 'Revenue' column)
- 1483 = Demand (from 'Demand' column)
- 10440 = Initial Inventory (from 'Initial Inventory' column)

The effective upper bound is $\min\{1483, 10440\} = 1483$.