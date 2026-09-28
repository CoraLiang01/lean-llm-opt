##### Decision Variables

Let $x_i$ = number of units of product $i$ (among ‘27in’ products) to fulfill, for each $i$ in the set of ‘27in’ products.

##### Parameters

- $P = \{$
    1: "27in 4K Gaming Monitor", 
    2: "27in FHD Monitor"
  $\}$
- Revenue per unit:
  - $r_1 = 389.99$ (for "27in 4K Gaming Monitor")
  - $r_2 = 149.99$ (for "27in FHD Monitor")
- Demand:
  - $d_1 = 12474$
  - $d_2 = 15057$
- Initial Inventory:
  - $I_1 = 62440$
  - $I_2 = 75500$

##### Mathematical Model

$\max\ 389.99\,x_1 + 149.99\,x_2$

subject to

$\begin{align*}
& 0 \leq x_1 \leq \min\{12474,\ 62440\} \\
& 0 \leq x_2 \leq \min\{15057,\ 75500\} \\
& x_1,\ x_2 \text{ continuous (or integer, if required)}
\end{align*}$

Or, explicitly:

$\begin{align*}
& 0 \leq x_1 \leq 12474 \\
& 0 \leq x_2 \leq 15057 \\
\end{align*}$

where
- $x_1$: units of "27in 4K Gaming Monitor" fulfilled
- $x_2$: units of "27in FHD Monitor" fulfilled

All coefficients and identifiers are as retrieved.