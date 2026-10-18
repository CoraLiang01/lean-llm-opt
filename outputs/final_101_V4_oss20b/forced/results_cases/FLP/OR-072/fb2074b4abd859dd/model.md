##### Decision Variables

$x_t \geq 0$: number of drivers who start their 4‑hour shift at the beginning of time period $t$, $t\in T=\{1,2,\dots ,24\}$ (integer).

##### Objective Function

$\displaystyle \min \sum_{t\in T} x_t$

##### Constraints

1. **Coverage of each time period**  
   For every period $j\in T$  
   $\displaystyle \sum_{t=\max \{1,j-3\}}^{j} x_t \;\geq\; d_j,\quad \forall j\in T$

2. **Non‑negativity and integrality**  
   $\displaystyle x_t \in \mathbb{Z}_{+},\quad \forall t\in T$

##### Parameters

- Demand vector $d_j$ (drivers required in each 1‑hour period):

  $d=\bigl[20,\;18,\;15,\;15,\;20,\;30,\;60,\;70,\;50,\;55,\;65,\;75,\;80,\;70,\;60,\;55,\;60,\;75,\;85,\;70,\;50,\;40,\;35,\;25\bigr]$

- Set of periods $T=\{1,2,\dots ,24\}$.

- Upper bound on total drivers (optional):  
  $M=\sum_{j=1}^{24} d_j = 1280$.

The matrix $A$ that encodes the coverage constraints is a $24\times24$ binary matrix where $A_{jt}=1$ if a driver starting at period $t$ works during period $j$ (i.e., $t\le j\le t+3$) and $0$ otherwise.  The constraints can be written compactly as $A\,x \ge d$.