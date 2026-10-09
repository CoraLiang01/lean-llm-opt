##### Sets and Indices

- $T = \{1,2,3,4\}$: time periods
- $K = \{1,2,\ldots,10\}$: trucks (truck_id from 1 to 10)

##### Parameters (from parameters.csv, source order)

For each truck $k$:
- $Q_k$: maximum capacity
- $S_k$: startup cost
- $C_k$: unit transportation cost

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
   $$
   \sum_{k=1}^{10} x_{k,t} \leq 0.9 \sum_{k=1}^{10} Q_k y_{k,t} \qquad \forall t=1,2,3,4
   $$

2. **Truck capacity and activity:**
   $$
   0 \leq x_{k,t} \leq Q_k y_{k,t} \qquad \forall k=1,\ldots,10;\ t=1,2,3,4
   $$

3. **Startup logic:**
   - All trucks are initially off:
     $$
     y_{k,0} = 0 \qquad \forall k
     $$
   - Startup variable definition:
     $$
     u_{k,t} \geq y_{k,t} - y_{k,t-1} \qquad \forall k,\ t=1,2,3,4
     $$
     $$
     u_{k,t} \leq 1 - y_{k,t-1} \qquad \forall k,\ t=1,2,3,4
     $$
     $$
     u_{k,t} \leq y_{k,t} \qquad \forall k,\ t=1,2,3,4
     $$
   - No startups allowed in period 4:
     $$
     u_{k,4} = 0 \qquad \forall k
     $$

4. **Minimum up-time (if started, must stay on at least 2 periods):**
   $$
   y_{k,t+1} \geq u_{k,t} \qquad \forall k,\ t=1,2,3
   $$

5. **Minimum down-time (if shut down, must stay off at least 2 periods):**
   $$
   y_{k,t-1} - y_{k,t} \leq 1 - y_{k,t+1} \qquad \forall k,\ t=2,3
   $$
   $$
   y_{k,t-1} - y_{k,t} \leq 1 - y_{k,t+2} \qquad \forall k,\ t=2
   $$
   (Alternatively, for $t=2,3$, if $y_{k,t-1}=1$ and $y_{k,t}=0$, then $y_{k,t+1}=0$ and $y_{k,t+2}=0$.)

   More precisely, for $t=2,3$:
   $$
   y_{k,t-1} - y_{k,t} \leq 1 - y_{k,t+1} \\
   y_{k,t-1} - y_{k,t} \leq 1 - y_{k,t+2}
   $$
   For $t=3$, $y_{k,5}$ is undefined, so only $y_{k,4}$ is enforced.

6. **No restart before two periods after shutdown:**
   $$
   u_{k,t} + y_{k,t-1} - y_{k,t-2} \leq 1 \qquad \forall k,\ t=3,4
   $$
   (If $y_{k,t-2}=1$, $y_{k,t-1}=0$, then $u_{k,t}=0$.)

7. **Load ramping (change in transported weight per truck per period $\leq 300$ kg):**
   $$
   |x_{k,t} - x_{k,t-1}| \leq 300 \qquad \forall k,\ t=2,3,4
   $$
   (Can be written as two inequalities:
   $$
   x_{k,t} - x_{k,t-1} \leq 300 \\
   x_{k,t-1} - x_{k,t} \leq 300
   $$)

   For $t=1$, define $x_{k,0}=0$.

8. **Inactive trucks transport zero:**
   $$
   x_{k,t} \leq Q_k y_{k,t} \qquad \forall k,\ t=1,2,3,4
   $$
   (Already included above.)

##### Variable domains

- $y_{k,t} \in \{0,1\}$
- $u_{k,t} \in \{0,1\}$
- $x_{k,t} \geq 0$

##### Initial conditions

- $y_{k,0} = 0$, $x_{k,0} = 0$ for all $k$

---

###### Retrieved Information

| truck_id | Q    | S   | C   |
|----------|------|-----|-----|
| 1        | 1000 | 500 | 2.0 |
| 2        | 800  | 300 | 3.0 |
| 3        | 1200 | 400 | 2.5 |
| 4        | 600  | 250 | 3.0 |
| 5        | 900  | 450 | 2.2 |
| 6        | 700  | 280 | 2.8 |
| 7        | 1100 | 420 | 2.4 |
| 8        | 500  | 200 | 3.2 |
| 9        | 1000 | 480 | 2.1 |
| 10       | 650  | 260 | 2.9 |

Demands: $d_1=1500$, $d_2=2000$, $d_3=1800$, $d_4=1000$