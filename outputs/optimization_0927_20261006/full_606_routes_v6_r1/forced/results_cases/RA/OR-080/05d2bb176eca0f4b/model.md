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
- $z_{k,t} \in \{0,1\}$: 1 if truck $k$ is started up in period $t$, 0 otherwise
- $x_{k,t} \geq 0$: weight transported by truck $k$ in period $t$ (kg)

Model:

Minimize total cost:
$$
\min \sum_{k \in K} \sum_{t \in T} \left( S_k z_{k,t} + C_k x_{k,t} \right)
$$

Subject to:

**1. Startup logic and initial state:**

- All trucks are initially off:
  $$
  y_{k,0} = 0 \quad \forall k \in K
  $$
  (We do not need to define $y_{k,0}$ as a variable; just use it as 0 in constraints.)

- Startup definition:
  $$
  z_{k,t} \geq y_{k,t} - y_{k,t-1} \quad \forall k \in K, \; t \in T
  $$
  $$
  z_{k,t} \leq 1 - y_{k,t-1} \quad \forall k \in K, \; t \in T
  $$
  $$
  z_{k,t} \leq y_{k,t} \quad \forall k \in K, \; t \in T
  $$
  (Alternatively, $z_{k,t} = 1$ iff $y_{k,t}=1$ and $y_{k,t-1}=0$.)

- No startup allowed in period 4:
  $$
  z_{k,4} = 0 \quad \forall k \in K
  $$

**2. Minimum up-time (if started, must stay on at least 2 periods):**

For $t=1,2,3$:
$$
y_{k,t+1} \geq z_{k,t} \quad \forall k \in K, \; t=1,2,3
$$

**3. Minimum down-time (if shut down, must stay off for 2 periods):**

For $t=1,2$:
$$
y_{k,t} - y_{k,t+1} \leq 1 - y_{k,t+2} \quad \forall k \in K
$$
(If $y_{k,t}=1$ and $y_{k,t+1}=0$, then $y_{k,t+2}=0$.)

**4. Truck activity and load bounds:**

- Inactive trucks transport zero:
  $$
  x_{k,t} \leq Q_k y_{k,t} \quad \forall k \in K, \; t \in T
  $$
  $$
  x_{k,t} \geq 0 \quad \forall k \in K, \; t \in T
  $$

**5. Load ramping constraints (change in load per truck per period ≤ 300 kg):**

For $t=1$:
$$
|x_{k,1} - 0| \leq 300 \quad \forall k \in K
$$

For $t=2,3,4$:
$$
|x_{k,t} - x_{k,t-1}| \leq 300 \quad \forall k \in K
$$

(Linearize with two inequalities for each $k,t$:
$$
x_{k,t} - x_{k,t-1} \leq 300
$$
$$
x_{k,t-1} - x_{k,t} \leq 300
$$
with $x_{k,0} = 0$.)

**6. Demand satisfaction and spare capacity:**

For each $t \in T$:
$$
\sum_{k \in K} x_{k,t} \geq d_t
$$
$$
\sum_{k \in K} x_{k,t} \leq 0.9 \sum_{k \in K} Q_k y_{k,t}
$$

**7. Variable domains:**

$$
y_{k,t} \in \{0,1\} \quad \forall k \in K, \; t \in T
$$
$$
z_{k,t} \in \{0,1\} \quad \forall k \in K, \; t \in T
$$
$$
x_{k,t} \geq 0 \quad \forall k \in K, \; t \in T
$$

**Parameter values (from parameters.csv, in source order):**

- $K = \{1,2,3,4,5,6,7,8,9,10\}$
- $Q_1=1000$, $Q_2=800$, $Q_3=1200$, $Q_4=600$, $Q_5=900$, $Q_6=700$, $Q_7=1100$, $Q_8=500$, $Q_9=1000$, $Q_{10}=650$
- $S_1=500$, $S_2=300$, $S_3=400$, $S_4=250$, $S_5=450$, $S_6=280$, $S_7=420$, $S_8=200$, $S_9=480$, $S_{10}=260$
- $C_1=2.0$, $C_2=3.0$, $C_3=2.5$, $C_4=3.0$, $C_5=2.2$, $C_6=2.8$, $C_7=2.4$, $C_8=3.2$, $C_9=2.1$, $C_{10}=2.9$
- $d_1=1500$, $d_2=2000$, $d_3=1800$, $d_4=1000$

**All indices, coefficients, and constraints are as above, using the truck_id and parameter values in the order and with the identifiers as retrieved.**