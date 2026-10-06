##### Sets
- $T = \{1,2,3,4\}$: periods
- $K = \{1,2,\ldots,10\}$: trucks

##### Parameters (from parameters.csv, source order)
For each truck $k$:
- $Q_k$: maximum capacity (kg)
- $S_k$: startup cost
- $C_k$: unit transportation cost (per kg)

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
- $u_{k,t} \in \{0,1\}$: 1 if truck $k$ is started up at the beginning of period $t$, 0 otherwise
- $x_{k,t} \geq 0$: weight transported by truck $k$ in period $t$ (kg)

##### Objective
Minimize total startup and transportation costs:
$$
\min \sum_{k=1}^{10} \sum_{t=1}^4 S_k u_{k,t} + \sum_{k=1}^{10} \sum_{t=1}^4 C_k x_{k,t}
$$

##### Constraints

1. **Demand satisfaction (per period):**
   $$
   \sum_{k=1}^{10} x_{k,t} \geq d_t \qquad \forall t \in T
   $$

2. **Spare-capacity buffer (per period):**
   $$
   \sum_{k=1}^{10} x_{k,t} \leq 0.9 \sum_{k=1}^{10} Q_k y_{k,t} \qquad \forall t \in T
   $$

3. **Truck capacity and activity:**
   $$
   0 \leq x_{k,t} \leq Q_k y_{k,t} \qquad \forall k \in K,\, t \in T
   $$

4. **Startup logic (initial state and transitions):**
   - All trucks are initially off:
     $$
     y_{k,0} = 0 \qquad \forall k \in K
     $$
   - Startup variable definition:
     $$
     u_{k,t} \geq y_{k,t} - y_{k,t-1} \qquad \forall k \in K,\, t=1,\ldots,4
     $$
     $$
     u_{k,t} \leq 1 - y_{k,t-1} \qquad \forall k \in K,\, t=1,\ldots,4
     $$
     $$
     u_{k,t} \leq y_{k,t} \qquad \forall k \in K,\, t=1,\ldots,4
     $$
   - No startups allowed in period 4:
     $$
     u_{k,4} = 0 \qquad \forall k \in K
     $$

5. **Minimum up-time (once started, must stay on at least 2 periods):**
   $$
   y_{k,t+1} \geq y_{k,t} - u_{k,t} \qquad \forall k \in K,\, t=1,2,3
   $$
   (If started at $t$, must be on at $t+1$.)

6. **Minimum down-time (if shut down, must stay off for 2 periods):**
   $$
   y_{k,t-1} - y_{k,t} \leq 1 - y_{k,t+1} \qquad \forall k \in K,\, t=2,3
   $$
   $$
   y_{k,t-1} - y_{k,t} \leq 1 - y_{k,t+2} \qquad \forall k \in K,\, t=2
   $$
   (If shut down at $t$, must be off at $t+1$ and $t+2$.)

7. **Weight change limit (per truck, adjacent periods):**
   $$
   |x_{k,t} - x_{k,t-1}| \leq 300 \qquad \forall k \in K,\, t=1,\ldots,4
   $$
   (Define $x_{k,0} = 0$.)

   This can be linearized as:
   $$
   x_{k,t} - x_{k,t-1} \leq 300 \qquad \forall k \in K,\, t=1,\ldots,4
   $$
   $$
   x_{k,t-1} - x_{k,t} \leq 300 \qquad \forall k \in K,\, t=1,\ldots,4
   $$

8. **Inactive trucks transport zero:**
   $$
   x_{k,t} \leq Q_k y_{k,t} \qquad \forall k \in K,\, t \in T
   $$
   (Already included above.)

##### Variable domains
- $y_{k,t} \in \{0,1\}$
- $u_{k,t} \in \{0,1\}$
- $x_{k,t} \geq 0$

##### Initial conditions
- $y_{k,0} = 0$, $x_{k,0} = 0$ for all $k$

##### Data (from parameters.csv, source order)
- $Q_1=1000$, $S_1=500$, $C_1=2.0$
- $Q_2=800$, $S_2=300$, $C_2=3.0$
- $Q_3=1200$, $S_3=400$, $C_3=2.5$
- $Q_4=600$, $S_4=250$, $C_4=3.0$
- $Q_5=900$, $S_5=450$, $C_5=2.2$
- $Q_6=700$, $S_6=280$, $C_6=2.8$
- $Q_7=1100$, $S_7=420$, $C_7=2.4$
- $Q_8=500$, $S_8=200$, $C_8=3.2$
- $Q_9=1000$, $S_9=480$, $C_9=2.1$
- $Q_{10}=650$, $S_{10}=260$, $C_{10}=2.9$

Demands:
- $d_1=1500$, $d_2=2000$, $d_3=1800$, $d_4=1000$

##### Summary of variables
- $x_{k,t}$: continuous, $\geq 0$
- $y_{k,t}$: binary
- $u_{k,t}$: binary

All indices, coefficients, and constraints are preserved in source order.