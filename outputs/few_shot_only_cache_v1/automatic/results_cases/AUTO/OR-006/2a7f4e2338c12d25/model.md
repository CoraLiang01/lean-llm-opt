##### Decision Variables

Let $x_{ij} \geq 0$ be the quantity shipped from warehouse $i$ to store $j$, for all $i \in I$, $j \in J$ (continuous).

##### Sets

- Warehouses $I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}, \text{S5}, \text{S6}, \text{S7}, \text{S8}, \text{S9}, \text{S10}\}$
- Stores $J = \{\text{C1}, \text{C2}, \text{C3}, \text{C4}, \text{C5}, \text{C6}, \text{C7}, \text{C8}, \text{C9}, \text{C10}\}$

##### Parameters

- Demand for each store $j$:
  - $d_{\text{C1}} = 45$
  - $d_{\text{C2}} = 23$
  - $d_{\text{C3}} = 94$
  - $d_{\text{C4}} = 92$
  - $d_{\text{C5}} = 57$
  - $d_{\text{C6}} = 52$
  - $d_{\text{C7}} = 23$
  - $d_{\text{C8}} = 99$
  - $d_{\text{C9}} = 99$
  - $d_{\text{C10}} = 77$

- Supply capacity for each warehouse $i$:
  - $s_{\text{S1}} = 127$
  - $s_{\text{S2}} = 236$
  - $s_{\text{S3}} = 168$
  - $s_{\text{S4}} = 115$
  - $s_{\text{S5}} = 280$
  - $s_{\text{S6}} = 179$
  - $s_{\text{S7}} = 135$
  - $s_{\text{S8}} = 263$
  - $s_{\text{S9}} = 283$
  - $s_{\text{S10}} = 476$

- Transportation cost per unit from warehouse $i$ to store $j$ ($c_{ij}$):

\[
\begin{array}{c|cccccccccc}
 & \text{C1} & \text{C2} & \text{C3} & \text{C4} & \text{C5} & \text{C6} & \text{C7} & \text{C8} & \text{C9} & \text{C10} \\
\hline
\text{S1} & 2077.058672521021 & 0.0 & 54.33526480458508 & 0.0 & 0.0 & 36.17284629162332 & 0.0 & 0.0 & 169.33026926588778 & 0.0 \\
\text{S2} & 2077.058672521021 & 0.0 & 1141.0405608962865 & 0.0 & 0.0 & 651.1112332492198 & 0.0 & 0.0 & 8.063346155518467 & 0.0 \\
\text{S3} & 79.9210295982608 & 474.24509131006675 & 1477.0676289106607 & 22.583099586193658 & 474.24509131006675 & 41.106596962251096 & 474.24509131006675 & 474.24509131006675 & 624.162539502301 & 474.24509131006675 \\
\text{S4} & 1659.336929105112 & 57.20541468776147 & 186.1519048103841 & 1201.3137084429907 & 1029.6974643797064 & 41.82210594495074 & 57.20541468776147 & 1201.3137084429907 & 884.5633870657458 & 1029.6974643797064 \\
\text{S5} & 1297.2567040858307 & 77.76629131320436 & 24.26760227579214 & 1399.7932436376784 & 77.76629131320436 & 53.91161728496604 & 1399.7932436376784 & 77.76629131320436 & 1255.115148013589 & 1399.7932436376784 \\
\text{S6} & 1998.9090658724567 & 985.3165435695341 & 2.8541686885814643 & 1149.53596749779 & 985.3165435695341 & 730.6923647662475 & 54.73980797608523 & 985.3165435695341 & 46.803102206463265 & 1149.53596749779 \\
\text{S7} & 1780.3360050180179 & 0.0 & 1141.0405608962865 & 0.0 & 0.0 & 36.17284629162332 & 0.0 & 0.0 & 8.063346155518467 & 0.0 \\
\text{S8} & 75.40935896233042 & 1338.1987290721909 & 21.391345987195333 & 74.34437383734394 & 74.34437383734394 & 937.3506239061463 & 1338.1987290721909 & 1338.1987290721909 & 1392.1186581383768 & 1338.1987290721909 \\
\text{S9} & 98.90755583433433 & 0.0 & 978.0347664825314 & 0.0 & 0.0 & 651.1112332492198 & 0.0 & 0.0 & 169.33026926588778 & 0.0 \\
\text{S10} & 2077.058672521021 & 0.0 & 54.33526480458508 & 0.0 & 0.0 & 36.17284629162332 & 0.0 & 0.0 & 145.1402307993324 & 0.0 \\
\end{array}
\]

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

##### Constraints

1. **Demand satisfaction:** For each store $j \in J$,
   \[
   \sum_{i \in I} x_{ij} \geq d_j
   \]
   That is,
   \begin{align*}
   x_{\text{S1},\text{C1}} + x_{\text{S2},\text{C1}} + \cdots + x_{\text{S10},\text{C1}} &\geq 45 \\
   x_{\text{S1},\text{C2}} + x_{\text{S2},\text{C2}} + \cdots + x_{\text{S10},\text{C2}} &\geq 23 \\
   &\vdots \\
   x_{\text{S1},\text{C10}} + x_{\text{S2},\text{C10}} + \cdots + x_{\text{S10},\text{C10}} &\geq 77 \\
   \end{align*}

2. **Supply capacity:** For each warehouse $i \in I$,
   \[
   \sum_{j \in J} x_{ij} \leq s_i
   \]
   That is,
   \begin{align*}
   x_{\text{S1},\text{C1}} + x_{\text{S1},\text{C2}} + \cdots + x_{\text{S1},\text{C10}} &\leq 127 \\
   x_{\text{S2},\text{C1}} + x_{\text{S2},\text{C2}} + \cdots + x_{\text{S2},\text{C10}} &\leq 236 \\
   &\vdots \\
   x_{\text{S10},\text{C1}} + x_{\text{S10},\text{C2}} + \cdots + x_{\text{S10},\text{C10}} &\leq 476 \\
   \end{align*}

3. **Non-negativity:** For all $i \in I$, $j \in J$,
   \[
   x_{ij} \geq 0
   \]

##### Complete Model

\[
\begin{align*}
\min\ & \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} \\
\text{s.t.}\quad
& \sum_{i \in I} x_{ij} \geq d_j \quad \forall j \in J \\
& \sum_{j \in J} x_{ij} \leq s_i \quad \forall i \in I \\
& x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
\end{align*}
\]

where all sets, parameters, and coefficients are as listed above.