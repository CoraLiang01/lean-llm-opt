##### Sets and Indices

- $I = \{1,2,\ldots,10\}$: set of trucks, indexed by $i$
- $T = \{1,2,3,4\}$: set of periods, indexed by $t$

##### Parameters (from parameters.csv)

For each truck $i$:

| $i$ | $Q_i$ | $S_i$ | $C_i$ |
|---|------|------|------|
| 1 | 1000 | 500  | 2.0  |
| 2 | 800  | 300  | 3.0  |
| 3 | 1200 | 400  | 2.5  |
| 4 | 600  | 250  | 3.0  |
| 5 | 900  | 450  | 2.2  |
| 6 | 700  | 280  | 2.8  |
| 7 | 1100 | 420  | 2.4  |
| 8 | 500  | 200  | 3.2  |
| 9 | 1000 | 480  | 2.1  |
|10 | 650  | 260  | 2.9  |

Customer demand per period:

- $d_1 = 1500$
- $d_2 = 2000$
- $d_3 = 1800$
- $d_4 = 1000$

##### Decision Variables

- $y_{i,t} \in \{0,1\}$: 1 if truck $i$ is active in period $t$, 0 otherwise
- $u_{i,t} \in \{0,1\}$: 1 if truck $i$ is started up in period $t$, 0 otherwise
- $z_{i,t} \in \{0,1\}$: 1 if truck $i$ is shut down in period $t$, 0 otherwise
- $x_{i,t} \geq 0$: weight transported by truck $i$ in period $t$ (kg)

##### Objective Function

Minimize total startup and transportation costs:
$$
\min \sum_{i=1}^{10} \sum_{t=1}^{4} S_i u_{i,t} + \sum_{i=1}^{10} \sum_{t=1}^{4} C_i x_{i,t}
$$

##### Constraints

1. **Startup and shutdown logic** (initially off):

   For all $i$ and $t=1$:
   $$
   y_{i,1} = u_{i,1}
   $$
   For all $i$ and $t=2,3,4$:
   $$
   y_{i,t} - y_{i,t-1} = u_{i,t} - z_{i,t}
   $$

2. **Minimum up-time (at least 2 consecutive periods if started):**

   For all $i$ and $t=1,2,3$:
   $$
   y_{i,t+1} \geq u_{i,t}
   $$
   (If started in $t$, must be on in $t+1$.)

   No startup allowed in period 4:
   $$
   u_{i,4} = 0 \quad \forall i
   $$

3. **Minimum down-time (if shut down, must stay off for 2 periods):**

   For all $i$ and $t=1,2$:
   $$
   y_{i,t+1} \leq 1 - z_{i,t}
   $$
   $$
   y_{i,t+2} \leq 1 - z_{i,t}
   $$
   (If shut down in $t$, must be off in $t+1$ and $t+2$.)

   For $t=3$:
   $$
   y_{i,4} \leq 1 - z_{i,3}
   $$
   (No $t=5$.)

4. **Inactive trucks transport zero:**

   For all $i,t$:
   $$
   x_{i,t} \leq Q_i y_{i,t}
   $$

5. **Truck capacity:**

   For all $i,t$:
   $$
   0 \leq x_{i,t} \leq Q_i y_{i,t}
   $$

6. **Load change limit (max 300 kg between adjacent periods):**

   For all $i$ and $t=2,3,4$:
   $$
   x_{i,t} - x_{i,t-1} \leq 300
   $$
   $$
   x_{i,t-1} - x_{i,t} \leq 300
   $$

   For $t=1$, define $x_{i,0}=0$.

7. **Demand satisfaction:**

   For all $t$:
   $$
   \sum_{i=1}^{10} x_{i,t} \geq d_t
   $$

8. **Spare-capacity buffer (total load $\leq$ 90% of active capacity):**

   For all $t$:
   $$
   \sum_{i=1}^{10} x_{i,t} \leq 0.9 \sum_{i=1}^{10} Q_i y_{i,t}
   $$

9. **Variable domains:**

   $$
   y_{i,t} \in \{0,1\},\quad u_{i,t} \in \{0,1\},\quad z_{i,t} \in \{0,1\},\quad x_{i,t} \geq 0
   $$

##### Data Table

| $i$ | $Q_i$ | $S_i$ | $C_i$ |
|---|------|------|------|
| 1 | 1000 | 500  | 2.0  |
| 2 | 800  | 300  | 3.0  |
| 3 | 1200 | 400  | 2.5  |
| 4 | 600  | 250  | 3.0  |
| 5 | 900  | 450  | 2.2  |
| 6 | 700  | 280  | 2.8  |
| 7 | 1100 | 420  | 2.4  |
| 8 | 500  | 200  | 3.2  |
| 9 | 1000 | 480  | 2.1  |
|10 | 650  | 260  | 2.9  |

Demands: $d_1=1500$, $d_2=2000$, $d_3=1800$, $d_4=1000$.

##### Summary

Minimize total startup and transportation costs, subject to truck activation, minimum up/down time, load change, demand, and spare-capacity constraints, using the parameters above.