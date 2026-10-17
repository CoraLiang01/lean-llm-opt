##### Sets and Indices

- $T = \{1,2,3,4\}$: periods
- $K = \{1,2,3,4,5,6,7,8,9,10\}$: trucks

##### Parameters (from parameters.csv, source order)

For each truck $k$:

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

Demands:
- $d_1 = 1500$
- $d_2 = 2000$
- $d_3 = 1800$
- $d_4 = 1000$

##### Decision Variables

- $y_{k,t} \in \{0,1\}$: 1 if truck $k$ is active in period $t$, 0 otherwise
- $u_{k,t} \in \{0,1\}$: 1 if truck $k$ is started up in period $t$, 0 otherwise
- $x_{k,t} \geq 0$: weight transported by truck $k$ in period $t$ (kg)

##### Objective

Minimize total startup and transportation costs:
$$
\min \sum_{k=1}^{10} \sum_{t=1}^4 S_k u_{k,t} + \sum_{k=1}^{10} \sum_{t=1}^4 C_k x_{k,t}
$$

##### Constraints

1. **Demand satisfaction (with buffer):**
   $$
   \sum_{k=1}^{10} x_{k,t} \geq d_t \qquad \forall t=1,2,3,4
   $$
2. **Spare-capacity buffer (no more than 90% of active capacity):**
   $$
   \sum_{k=1}^{10} x_{k,t} \leq 0.9 \sum_{k=1}^{10} Q_k y_{k,t} \qquad \forall t=1,2,3,4
   $$
3. **Truck capacity and activity:**
   $$
   0 \leq x_{k,t} \leq Q_k y_{k,t} \qquad \forall k=1,\ldots,10;\ t=1,2,3,4
   $$
4. **Startup logic:**
   - All trucks are initially off:
     $$
     y_{k,0} = 0 \qquad \forall k
     $$
   - Startup variable definition:
     $$
     u_{k,t} \geq y_{k,t} - y_{k,t-1} \qquad \forall k;\ t=1,2,3,4
     $$
     $$
     u_{k,t} \geq 0 \qquad \forall k;\ t=1,2,3,4
     $$
     $$
     u_{k,t} \leq 1 \qquad \forall k;\ t=1,2,3,4
     $$
   - No startups allowed in period 4:
     $$
     u_{k,4} = 0 \qquad \forall k
     $$
5. **Minimum up-time (if started, must stay on at least 2 periods):**
   $$
   y_{k,t+1} \geq y_{k,t} - u_{k,t} \qquad \forall k;\ t=1,2,3
   $$
   (If started at $t$, must be on at $t+1$)
6. **Minimum down-time (if shut down, must stay off at least 2 periods):**
   $$
   y_{k,t} + y_{k,t+1} \leq 1 + y_{k,t-1} \qquad \forall k;\ t=1,2,3
   $$
   (If $y_{k,t-1}=1$ and $y_{k,t}=0$, then $y_{k,t+1}=0$)
   - Cannot be restarted before $t+2$:
     $$
     y_{k,t+2} \leq y_{k,t} + y_{k,t+1} \qquad \forall k;\ t=1,2
     $$
7. **No startups in period 4:**
   $$
   u_{k,4} = 0 \qquad \forall k
   $$
8. **Inactive trucks transport zero:**
   $$
   x_{k,t} \leq Q_k y_{k,t} \qquad \forall k;\ t=1,2,3,4
   $$
   (already included above)
9. **Load change limit (including transitions to/from zero):**
   $$
   |x_{k,t} - x_{k,t-1}| \leq 300 \qquad \forall k;\ t=1,2,3,4
   $$
   where $x_{k,0} = 0$ for all $k$.

   This can be linearized as:
   $$
   x_{k,t} - x_{k,t-1} \leq 300 \qquad \forall k;\ t=1,2,3,4
   $$
   $$
   x_{k,t-1} - x_{k,t} \leq 300 \qquad \forall k;\ t=1,2,3,4
   $$
   with $x_{k,0}=0$.

##### Variable domains

- $y_{k,t} \in \{0,1\}$
- $u_{k,t} \in \{0,1\}$
- $x_{k,t} \geq 0$

##### Data (source order)

- $Q_k$: truck $k$'s max capacity (kg)
- $S_k$: truck $k$'s startup cost
- $C_k$: truck $k$'s unit transportation cost
- $d_t$: demand in period $t$

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

Demands:
- $d_1 = 1500$
- $d_2 = 2000$
- $d_3 = 1800$
- $d_4 = 1000$

##### Initial conditions

- $y_{k,0} = 0$ for all $k$
- $x_{k,0} = 0$ for all $k$

All constraints and coefficients are included as required by the query and retrieved data.