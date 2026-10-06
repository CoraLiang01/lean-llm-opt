##### Sets and Indices

- $I = \{1,2,\ldots,10\}$: set of trucks (truck_id from 1 to 10)
- $T = \{1,2,3,4\}$: set of time periods

##### Parameters (from parameters.csv)

| truck_id | $Q_i$ | $S_i$ | $C_i$ |
|----------|-------|-------|-------|
| 1        | 1000  | 500   | 2.0   |
| 2        | 800   | 300   | 3.0   |
| 3        | 1200  | 400   | 2.5   |
| 4        | 600   | 250   | 3.0   |
| 5        | 900   | 450   | 2.2   |
| 6        | 700   | 280   | 2.8   |
| 7        | 1100  | 420   | 2.4   |
| 8        | 500   | 200   | 3.2   |
| 9        | 1000  | 480   | 2.1   |
| 10       | 650   | 260   | 2.9   |

- $d_1 = 1500$, $d_2 = 2000$, $d_3 = 1800$, $d_4 = 1000$

##### Decision Variables

- $y_{i,t} \in \{0,1\}$: 1 if truck $i$ is active in period $t$, 0 otherwise
- $u_{i,t} \in \{0,1\}$: 1 if truck $i$ is started up at the beginning of period $t$, 0 otherwise
- $z_{i,t} \in \{0,1\}$: 1 if truck $i$ is shut down at the end of period $t$, 0 otherwise
- $x_{i,t} \geq 0$: weight transported by truck $i$ in period $t$ (continuous, in kg)

##### Objective Function

Minimize total startup and transportation costs:
$$
\min \sum_{i=1}^{10} \sum_{t=1}^{4} S_i u_{i,t} + \sum_{i=1}^{10} \sum_{t=1}^{4} C_i x_{i,t}
$$

##### Constraints

1. **Startup and shutdown logic** (initially off, startup and shutdown tracking):
   - All trucks are initially off:
     $$
     y_{i,0} = 0 \quad \forall i
     $$
     (Introduce $y_{i,0}$ as a fixed parameter for logic; not a variable.)

   - Truck activation state:
     $$
     y_{i,t} - y_{i,t-1} = u_{i,t} - z_{i,t} \quad \forall i, \forall t=1,2,3,4
     $$
     (Set $y_{i,0}=0$.)

2. **Minimum up-time (once started, must stay on at least 2 periods; cannot start in period 4):**
   - If started in $t=1,2,3$, must be on in $t$ and $t+1$:
     $$
     y_{i,t+1} \geq u_{i,t} \quad \forall i, \forall t=1,2,3
     $$
   - No startup in period 4:
     $$
     u_{i,4} = 0 \quad \forall i
     $$

3. **Minimum down-time (if shut down, must stay off for 2 periods):**
   - If shut down at end of $t=1,2$, must be off in $t+1$ and $t+2$:
     $$
     y_{i,t+1} \leq 1 - z_{i,t} \\
     y_{i,t+2} \leq 1 - z_{i,t} \quad \forall i, \forall t=1,2
     $$
   - If shut down at end of $t=3$, must be off in $t=4$:
     $$
     y_{i,4} \leq 1 - z_{i,3} \quad \forall i
     $$
   - Cannot shut down in period 4 (no effect, but for completeness):
     $$
     z_{i,4} = 0 \quad \forall i
     $$

4. **Inactive trucks transport zero:**
   $$
   x_{i,t} \leq Q_i y_{i,t} \quad \forall i, t
   $$

5. **Truck capacity:**
   $$
   0 \leq x_{i,t} \leq Q_i y_{i,t} \quad \forall i, t
   $$

6. **Demand satisfaction:**
   $$
   \sum_{i=1}^{10} x_{i,t} \geq d_t \quad \forall t=1,2,3,4
   $$

7. **Spare capacity buffer (total load ≤ 90% of active capacity):**
   $$
   \sum_{i=1}^{10} x_{i,t} \leq 0.9 \sum_{i=1}^{10} Q_i y_{i,t} \quad \forall t=1,2,3,4
   $$

8. **Load change limit (≤ 300 kg between adjacent periods, including to/from zero):**
   - For $t=1$:
     $$
     |x_{i,1} - 0| \leq 300 \quad \forall i
     $$
   - For $t=2,3,4$:
     $$
     |x_{i,t} - x_{i,t-1}| \leq 300 \quad \forall i, t=2,3,4
     $$
   (Linearize with two inequalities for each absolute value.)

##### Variable Domains

- $y_{i,t} \in \{0,1\}$, $u_{i,t} \in \{0,1\}$, $z_{i,t} \in \{0,1\}$, $x_{i,t} \geq 0$ (continuous)

---

##### All Data Used

- $I = \{1,2,3,4,5,6,7,8,9,10\}$
- $T = \{1,2,3,4\}$
- $Q = [1000, 800, 1200, 600, 900, 700, 1100, 500, 1000, 650]$
- $S = [500, 300, 400, 250, 450, 280, 420, 200, 480, 260]$
- $C = [2.0, 3.0, 2.5, 3.0, 2.2, 2.8, 2.4, 3.2, 2.1, 2.9]$
- $d_1 = 1500$, $d_2 = 2000$, $d_3 = 1800$, $d_4 = 1000$

---

##### Complete Model

Minimize
$$
\sum_{i=1}^{10} \sum_{t=1}^{4} S_i u_{i,t} + \sum_{i=1}^{10} \sum_{t=1}^{4} C_i x_{i,t}
$$

Subject to, for all $i=1,\ldots,10$, $t=1,\ldots,4$:

- $y_{i,0} = 0$
- $y_{i,t} - y_{i,t-1} = u_{i,t} - z_{i,t}$
- $y_{i,t+1} \geq u_{i,t}$ for $t=1,2,3$
- $u_{i,4} = 0$
- $y_{i,t+1} \leq 1 - z_{i,t}$ and $y_{i,t+2} \leq 1 - z_{i,t}$ for $t=1,2$
- $y_{i,4} \leq 1 - z_{i,3}$
- $z_{i,4} = 0$
- $x_{i,t} \leq Q_i y_{i,t}$
- $0 \leq x_{i,t} \leq Q_i y_{i,t}$
- $\sum_{i=1}^{10} x_{i,t} \geq d_t$
- $\sum_{i=1}^{10} x_{i,t} \leq 0.9 \sum_{i=1}^{10} Q_i y_{i,t}$
- $|x_{i,1}| \leq 300$
- $|x_{i,t} - x_{i,t-1}| \leq 300$ for $t=2,3,4$
- $y_{i,t}, u_{i,t}, z_{i,t} \in \{0,1\}$, $x_{i,t} \geq 0$

All coefficients and identifiers are as in parameters.csv and the user query.