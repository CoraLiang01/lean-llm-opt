##### Sets

- $K = \{1,2,\ldots,10\}$: trucks (indexed by $k$ in source order)
- $T = \{1,2,3,4\}$: periods

##### Parameters (from parameters.csv, source order)

| $k$ | $Q_k$ (max capacity) | $S_k$ (startup cost) | $C_k$ (unit cost) |
|-----|---------------------|----------------------|-------------------|
| 1   | 1000                | 500                  | 2.0               |
| 2   | 800                 | 300                  | 3.0               |
| 3   | 1200                | 400                  | 2.5               |
| 4   | 600                 | 250                  | 3.0               |
| 5   | 900                 | 450                  | 2.2               |
| 6   | 700                 | 280                  | 2.8               |
| 7   | 1100                | 420                  | 2.4               |
| 8   | 500                 | 200                  | 3.2               |
| 9   | 1000                | 480                  | 2.1               |
| 10  | 650                 | 260                  | 2.9               |

Customer demands (identical in each row, so use once):

- $d_1 = 1500$
- $d_2 = 2000$
- $d_3 = 1800$
- $d_4 = 1000$

##### Decision Variables

- $y_{k,t} \in \{0,1\}$: 1 if truck $k$ is active in period $t$, 0 otherwise
- $z_{k,t} \in \{0,1\}$: 1 if truck $k$ is started up in period $t$, 0 otherwise
- $x_{k,t} \geq 0$: weight transported by truck $k$ in period $t$ (kg), continuous

##### Objective

Minimize total startup and transportation costs:
$$
\min \sum_{k=1}^{10}\sum_{t=1}^4 S_k z_{k,t} + \sum_{k=1}^{10}\sum_{t=1}^4 C_k x_{k,t}
$$

##### Constraints

1. **Startup logic:** (with $y_{k,0}=0$ for all $k$)
   $$
   z_{k,t} \geq y_{k,t} - y_{k,t-1} \qquad \forall k=1,\ldots,10;\ t=1,\ldots,4
   $$
   $$
   y_{k,0} = 0 \qquad \forall k
   $$

2. **Minimum up-time (at least 2 consecutive periods):**
   $$
   y_{k,t+1} \geq z_{k,t} \qquad \forall k=1,\ldots,10;\ t=1,2,3
   $$
   $$
   z_{k,4} = 0 \qquad \forall k=1,\ldots,10
   $$

3. **Minimum down-time (at least 2 consecutive periods):**
   $$
   y_{k,t+1} \leq y_{k,t} \qquad \forall k=1,\ldots,10;\ t=1,2,3
   $$

4. **Inactive trucks transport zero:**
   $$
   x_{k,t} \leq Q_k y_{k,t} \qquad \forall k=1,\ldots,10;\ t=1,\ldots,4
   $$

5. **Active trucks cannot exceed capacity:**
   $$
   x_{k,t} \leq Q_k \qquad \forall k=1,\ldots,10;\ t=1,\ldots,4
   $$

6. **Load change limit (including transitions to/from zero):**
   $$
   x_{k,0} = 0 \qquad \forall k=1,\ldots,10
   $$
   $$
   x_{k,t} - x_{k,t-1} \leq 300 \qquad \forall k=1,\ldots,10;\ t=2,3,4
   $$
   $$
   x_{k,t-1} - x_{k,t} \leq 300 \qquad \forall k=1,\ldots,10;\ t=2,3,4
   $$

7. **Demand satisfaction:**
   $$
   \sum_{k=1}^{10} x_{k,t} \geq d_t \qquad t=1,2,3,4
   $$

8. **Spare-capacity buffer (total load $\leq$ 90% of active capacity):**
   $$
   \sum_{k=1}^{10} x_{k,t} \leq 0.9 \sum_{k=1}^{10} Q_k y_{k,t} \qquad t=1,2,3,4
   $$

9. **Variable domains:**
   $$
   y_{k,t} \in \{0,1\},\quad z_{k,t} \in \{0,1\},\quad x_{k,t} \geq 0 \qquad \forall k=1,\ldots,10;\ t=1,\ldots,4
   $$

##### Parameter values (from parameters.csv, source order):

| $k$ | $Q_k$ | $S_k$ | $C_k$ |
|-----|-------|-------|-------|
| 1   | 1000  | 500   | 2.0   |
| 2   | 800   | 300   | 3.0   |
| 3   | 1200  | 400   | 2.5   |
| 4   | 600   | 250   | 3.0   |
| 5   | 900   | 450   | 2.2   |
| 6   | 700   | 280   | 2.8   |
| 7   | 1100  | 420   | 2.4   |
| 8   | 500   | 200   | 3.2   |
| 9   | 1000  | 480   | 2.1   |
| 10  | 650   | 260   | 2.9   |

Customer demands:

- $d_1 = 1500$
- $d_2 = 2000$
- $d_3 = 1800$
- $d_4 = 1000$