Let:
- $I = \{1,2,\ldots,10\}$ be the set of trucks (truck_id from parameters.csv).
- $T = \{1,2,3,4\}$ be the set of time periods.
- $Q_i$ = maximum capacity of truck $i$ (from Q column).
- $S_i$ = startup cost of truck $i$ (from S column).
- $C_i$ = unit transportation cost of truck $i$ (from C column).
- $d_t$ = customer demand in period $t$ (from d1, d2, d3, d4 columns; all trucks have the same demand per period).

Decision variables:
- $y_{i,t} \in \{0,1\}$: 1 if truck $i$ is active in period $t$, 0 otherwise.
- $u_{i,t} \in \{0,1\}$: 1 if truck $i$ is started up at the beginning of period $t$, 0 otherwise.
- $x_{i,t} \geq 0$: weight (kg) transported by truck $i$ in period $t$.

Objective:
\[
\min \sum_{i \in I} \sum_{t \in T} \left( S_i u_{i,t} + C_i x_{i,t} \right)
\]

Subject to:

**1. Startup logic and initial state:**
\[
y_{i,0} = 0 \quad \forall i \in I \qquad \text{(trucks initially off)}
\]
\[
u_{i,1} \geq y_{i,1} \quad \forall i \in I
\]
\[
u_{i,t} \geq y_{i,t} - y_{i,t-1} \quad \forall i \in I, \; t=2,3,4
\]
\[
u_{i,t} \leq 1 - y_{i,t-1} \quad \forall i \in I, \; t=1,2,3,4
\]
\[
u_{i,4} = 0 \quad \forall i \in I \qquad \text{(cannot start in period 4)}
\]

**2. Minimum up-time (if started, must stay on at least 2 periods):**
\[
y_{i,t+1} \geq y_{i,t} - u_{i,t} \quad \forall i \in I, \; t=1,2,3
\]
(If started at $t$, must be on at $t+1$.)

**3. Minimum down-time (if shut down, must stay off for 2 periods):**
\[
y_{i,t} + y_{i,t+1} \leq 1 + y_{i,t-1} \quad \forall i \in I, \; t=2,3
\]
(If $y_{i,t-1}=1$, $y_{i,t}=0$ implies $y_{i,t+1}=0$.)

Alternatively, for all $i$ and $t=2,3$:
\[
y_{i,t-1} - y_{i,t} \leq 1 - y_{i,t+1}
\]
(If truck is shut down at $t$, must be off at $t+1$.)

**4. Truck activity and load:**
\[
0 \leq x_{i,t} \leq Q_i y_{i,t} \quad \forall i \in I, \; t \in T
\]

**5. Load ramping (change in load per truck per period):**
\[
|x_{i,t} - x_{i,t-1}| \leq 300 \quad \forall i \in I, \; t=2,3,4
\]
Set $x_{i,0} = 0$ for all $i$.

**6. Demand satisfaction:**
\[
\sum_{i \in I} x_{i,t} \geq d_t \quad \forall t \in T
\]

**7. Spare capacity buffer (total load cannot exceed 90% of active capacity):**
\[
\sum_{i \in I} x_{i,t} \leq 0.9 \sum_{i \in I} Q_i y_{i,t} \quad \forall t \in T
\]

**8. Variable domains:**
\[
y_{i,t} \in \{0,1\}, \quad u_{i,t} \in \{0,1\}, \quad x_{i,t} \geq 0 \quad \forall i \in I, \; t \in T
\]

---

**Parameters from parameters.csv (source order):**

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

Demands:
- $d_1 = 1500$
- $d_2 = 2000$
- $d_3 = 1800$
- $d_4 = 1000$

All constraints and coefficients are as above, using the exact identifiers and values from the retrieved data.