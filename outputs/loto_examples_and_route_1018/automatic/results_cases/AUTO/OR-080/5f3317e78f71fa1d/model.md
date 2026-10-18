Let:

- $T = \{1,2,3,4\}$ (time periods)
- $K = \{1,2,3,4,5,6,7,8,9,10\}$ (truck_id, as in parameters.csv)
- $Q_k$ = maximum capacity of truck $k$
- $S_k$ = startup cost of truck $k$
- $C_k$ = unit transportation cost of truck $k$
- $d_t$ = customer demand in period $t$

Parameters (from parameters.csv, in source order):

| truck_id | $Q_k$ | $S_k$ | $C_k$ |
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

Demands:

- $d_1 = 1500$
- $d_2 = 2000$
- $d_3 = 1800$
- $d_4 = 1000$

Decision variables:

- $y_{k,t} \in \{0,1\}$: 1 if truck $k$ is active in period $t$, 0 otherwise
- $u_{k,t} \in \{0,1\}$: 1 if truck $k$ is started up at the beginning of period $t$, 0 otherwise
- $x_{k,t} \geq 0$: weight transported by truck $k$ in period $t$ (kg)

Model:

Minimize total cost:
$$
\min \sum_{k \in K} \sum_{t \in T} S_k u_{k,t} + \sum_{k \in K} \sum_{t \in T} C_k x_{k,t}
$$

Subject to:

**1. Startup logic and initial state:**

- All trucks are initially off:
  $$
  y_{k,0} = 0 \quad \forall k \in K
  $$
  (Introduce $y_{k,0}$ as a fixed parameter for startup logic.)

- Startup definition:
  $$
  u_{k,t} \geq y_{k,t} - y_{k,t-1} \quad \forall k \in K, \; t \in T
  $$
  $$
  u_{k,t} \geq 0 \quad \forall k \in K, \; t \in T
  $$
  $$
  u_{k,t} \leq 1 \quad \forall k \in K, \; t \in T
  $$
  (But $u_{k,t}$ is binary.)

- No startup allowed in period 4:
  $$
  u_{k,4} = 0 \quad \forall k \in K
  $$

**2. Minimum up-time (if started, must stay on at least 2 periods):**

- If started in $t=1,2,3$, must be on in $t$ and $t+1$:
  $$
  y_{k,t+1} \geq u_{k,t} \quad \forall k \in K, \; t=1,2,3
  $$

**3. Minimum down-time (if shut down, must stay off for 2 periods):**

- If truck is shut down in $t$ (i.e., $y_{k,t-1}=1$, $y_{k,t}=0$), then $y_{k,t+1}=0$ and $y_{k,t+2}=0$ (for $t=1,2$):
  $$
  y_{k,t-1} - y_{k,t} \leq 1 - y_{k,t+1} \quad \forall k \in K, \; t=1,2
  $$
  $$
  y_{k,t-1} - y_{k,t} \leq 1 - y_{k,t+2} \quad \forall k \in K, \; t=1,2
  $$
  (For $t=3$, $y_{k,5}$ is out of range, so only $y_{k,4}$ applies.)

- For $t=3$:
  $$
  y_{k,2} - y_{k,3} \leq 1 - y_{k,4} \quad \forall k \in K
  $$
  (No $y_{k,5}$.)

**4. Truck activity and load:**

- Inactive trucks transport zero:
  $$
  x_{k,t} \leq Q_k y_{k,t} \quad \forall k \in K, \; t \in T
  $$
  $$
  x_{k,t} \geq 0 \quad \forall k \in K, \; t \in T
  $$

- Active trucks cannot exceed capacity:
  $$
  x_{k,t} \leq Q_k y_{k,t} \quad \forall k \in K, \; t \in T
  $$

**5. Load ramping constraint:**

- For all $k \in K$, $t=2,3,4$:
  $$
  |x_{k,t} - x_{k,t-1}| \leq 300
  $$
  (Introduce auxiliary variables or split into two inequalities:)
  $$
  x_{k,t} - x_{k,t-1} \leq 300 \quad \forall k \in K, \; t=2,3,4
  $$
  $$
  x_{k,t-1} - x_{k,t} \leq 300 \quad \forall k \in K, \; t=2,3,4
  $$
  (Set $x_{k,0}=0$ for all $k$.)

**6. Demand satisfaction and spare capacity:**

- Total transported weight at least demand:
  $$
  \sum_{k \in K} x_{k,t} \geq d_t \quad \forall t \in T
  $$

- Total transported weight at most 90% of active capacity:
  $$
  \sum_{k \in K} x_{k,t} \leq 0.9 \sum_{k \in K} Q_k y_{k,t} \quad \forall t \in T
  $$

**7. Variable domains:**

- $y_{k,t} \in \{0,1\}$, $u_{k,t} \in \{0,1\}$, $x_{k,t} \geq 0$ for all $k \in K$, $t \in T$.

**8. Initial conditions:**

- $y_{k,0} = 0$, $x_{k,0} = 0$ for all $k \in K$.

---

**Numerical parameters (from parameters.csv, in source order):**

| truck_id | $Q_k$ | $S_k$ | $C_k$ |
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

Demands:

- $d_1 = 1500$
- $d_2 = 2000$
- $d_3 = 1800$
- $d_4 = 1000$

---

**Summary of variables:**

- $y_{k,t} \in \{0,1\}$: truck $k$ active in period $t$
- $u_{k,t} \in \{0,1\}$: truck $k$ started up at beginning of period $t$
- $x_{k,t} \geq 0$: weight transported by truck $k$ in period $t$ (kg)

All indices, parameters, and constraints are as above, using the exact identifiers and values from the retrieved data.