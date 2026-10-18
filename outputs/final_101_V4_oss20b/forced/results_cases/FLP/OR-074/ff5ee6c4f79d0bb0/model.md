##### Decision Variables

$x_s \in \mathbb{Z}_{\ge 0}$: number of waitstaff who start working at time slot $s$, $s\in S=\{1,2,\dots ,48\}$ (each slot represents a 30‑minute interval).

##### Objective Function

$\displaystyle \min \sum_{s\in S} x_s$

##### Constraints

1. **Coverage of demand at each time slot**  
   $\displaystyle \sum_{s=\max \{1,\,t-15\}}^{t} x_s \;\ge\; d_t ,\qquad \forall t\in S$

2. **Non‑negativity and integrality**  
   $\displaystyle x_s \;\ge\; 0 ,\; x_s \in \mathbb{Z},\qquad \forall s\in S$

##### Parameters

- $S=\{1,2,\dots ,48\}$ (48 half‑hour intervals covering 24 h).  
- Demand vector $d_t$ (required number of waitstaff at each interval):

| $t$ | Time Slot | $d_t$ |
|-----|-----------|-------|
| 1 | 2:00 am – 2:30 am | 2 |
| 2 | 2:30 am – 3:00 am | 3 |
| 3 | 3:00 am – 3:30 am | 4 |
| 4 | 3:30 am – 4:00 am | 6 |
| 5 | 4:00 am – 4:30 am | 5 |
| 6 | 4:30 am – 5:00 am | 4 |
| 7 | 5:00 am – 5:30 am | 5 |
| 8 | 5:30 am – 6:00 am | 6 |
| 9 | 6:00 am – 6:30 am | 7 |
| 10 | 6:30 am – 7:00 am | 8 |
| 11 | 7:00 am – 7:30 am | 9 |
| 12 | 7:30 am – 8:00 am | 9 |
| 13 | 8:00 am – 8:30 am | 8 |
| 14 | 8:30 am – 9:00 am | 8 |
| 15 | 9:00 am – 9:30 am | 9 |
| 16 | 9:30 am – 10:00 am | 9 |
| 17 | 10:00 am – 10:30 am | 10 |
| 18 | 10:30 am – 11:00 am | 12 |
| 19 | 11:00 am – 11:30 am | 11 |
| 20 | 11:30 am – 12:00 pm | 11 |
| 21 | 12:00 pm – 12:30 pm | 12 |
| 22 | 12:30 pm – 1:00 pm | 11 |
| 23 | 1:00 pm – 1:30 pm | 10 |
| 24 | 1:30 pm – 2:00 pm | 9 |
| 25 | 2:00 pm – 2:30 pm | 8 |
| 26 | 2:30 pm – 3:00 pm | 7 |
| 27 | 3:00 pm – 3:30 pm | 6 |
| 28 | 3:30 pm – 4:00 pm | 5 |
| 29 | 4:00 pm – 4:30 pm | 5 |
| 30 | 4:30 pm – 5:00 pm | 6 |
| 31 | 5:00 pm – 5:30 pm | 7 |
| 32 | 5:30 pm – 6:00 pm | 8 |
| 33 | 6:00 pm – 6:30 pm | 9 |
| 34 | 6:30 pm – 7:00 pm | 10 |
| 35 | 7:00 pm – 7:30 pm | 9 |
| 36 | 7:30 pm – 8:00 pm | 8 |
| 37 | 8:00 pm – 8:30 pm | 7 |
| 38 | 8:30 pm – 9:00 pm | 6 |
| 39 | 9:00 pm – 9:30 pm | 5 |
| 40 | 9:30 pm – 10:00 pm | 4 |
| 41 | 10:00 pm – 10:30 pm | 4 |
| 42 | 10:30 pm – 11:00 pm | 3 |
| 43 | 11:00 pm – 11:30 pm | 3 |
| 44 | 11:30 pm – 12:00 am | 3 |
| 45 | 12:00 am – 12:30 am | 3 |
| 46 | 12:30 am – 1:00 am | 4 |
| 47 | 1:00 am – 1:30 am | 4 |
| 48 | 1:30 am – 2:00 am | 4 |

- Coverage matrix $A\in\{0,1\}^{48\times48}$ where $A_{t,s}=1$ if a staff member who starts at slot $s$ works during slot $t$ (i.e., $s\le t\le s+15$), otherwise $A_{t,s}=0$.  
  For example, the first few rows of $A$ are:

  | $t$ | $A_{t,1}$ | $A_{t,2}$ | $A_{t,3}$ | … | $A_{t,48}$ |
  |-----|-----------|-----------|-----------|---|------------|
  | 1 | 1 | 0 | 0 | … | 0 |
  | 2 | 1 | 1 | 0 | … | 0 |
  | 3 | 1 | 1 | 1 | … | 0 |
  | … | … | … | … | … | … |
  | 16 | 1 | 1 | 1 | … | 0 |
  | 17 | 0 | 1 | 1 | … | 0 |
  | … | … | … | … | … | … |
  | 48 | 0 | 0 | 0 | … | 1 |

  (Each column contains 16 consecutive ones, starting at its column index and ending at column index + 15, truncated at 48.)

This linear integer program yields the minimum number of waitstaff required to satisfy the 24‑hour demand profile.