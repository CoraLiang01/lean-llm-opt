Let $x_i$ denote the number of units of product $i$ (where $i$ indexes the set of ‘27in’ products) to be fulfilled.

Sets:
- $i \in \{$"27in 4K Gaming Monitor", "27in FHD Monitor"$\}$

Parameters:
- $r_i$: Revenue per unit of product $i$
  - $r_{\text{"27in 4K Gaming Monitor"}} = 261.2933$
  - $r_{\text{"27in FHD Monitor"}} = 52.4965$
- $d_i$: Demand for product $i$
  - $d_{\text{"27in 4K Gaming Monitor"}} = 12474$
  - $d_{\text{"27in FHD Monitor"}} = 15057$
- $s_i$: Initial Inventory for product $i$
  - $s_{\text{"27in 4K Gaming Monitor"}} = 62440$
  - $s_{\text{"27in FHD Monitor"}} = 75500$

Decision Variables:
- $x_i \in \mathbb{Z}_{\geq 0}$, for all $i$ (number of units of product $i$ to fulfill)

Objective:
\[
\max \; 261.2933\, x_{\text{"27in 4K Gaming Monitor"}} + 52.4965\, x_{\text{"27in FHD Monitor"}}
\]

Subject to:
\[
\begin{align*}
& x_{\text{"27in 4K Gaming Monitor"}} \leq 12474 \\
& x_{\text{"27in FHD Monitor"}} \leq 15057 \\
& x_{\text{"27in 4K Gaming Monitor"}} \leq 62440 \\
& x_{\text{"27in FHD Monitor"}} \leq 75500 \\
& x_{\text{"27in 4K Gaming Monitor"}},\; x_{\text{"27in FHD Monitor"}} \in \mathbb{Z}_{\geq 0}
\end{align*}
\]

Where:
- The first two constraints ensure that fulfilled units do not exceed demand.
- The next two constraints ensure that fulfilled units do not exceed initial inventory.
- Variables are nonnegative integers.