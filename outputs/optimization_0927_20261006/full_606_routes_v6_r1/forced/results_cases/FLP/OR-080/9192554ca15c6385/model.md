##### Sets and Indices

- $I = \{1,2,\ldots,10\}$: set of trucks, indexed by $i$
- $T = \{1,2,3,4\}$: set of periods, indexed by $t$

##### Parameters

- $Q_i$: maximum capacity of truck $i$
- $S_i$: startup cost for truck $i$
- $C_i$: unit transportation cost for truck $i$
- $d_t$: customer demand in period $t$

From parameters.csv:

| $i$ | $Q_i$ | $S_i$ | $C_i$ |
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

- $y_{i,t} \in \{0,1\}$: 1 if truck $i$ is active in period $t$, 0 otherwise
- $u_{i,t} \in \{0,1\}$: 1 if truck $i$ is started up at the beginning of period $t$, 0 otherwise
- $z_{i,t} \in \{0,1\}$: 1 if truck $i$ is shut down at the end of period $t$, 0 otherwise
- $x_{i,t} \geq 0$: weight transported by truck $i$ in period $t$ (continuous, in kg)

##### Objective Function

\[
\min \sum_{i \in I} \sum_{t \in T} \left( S_i u_{i,t} + C_i x_{i,t} \right)
\]

##### Constraints

1. **Startup and shutdown logic:**
   - Initial state: all trucks are off before period 1.
     \[
     y_{i,0} = 0 \quad \forall i \in I
     \]
   - Truck activity transitions:
     \[
     y_{i,t} - y_{i,t-1} = u_{i,t} - z_{i,t} \quad \forall i \in I, \; t=1,\ldots,4
     \]
     (Define $y_{i,0}=0$.)

2. **Minimum up-time (once started, must stay on at least 2 periods):**
   - If started at $t$, must be on at $t$ and $t+1$ (for $t=1,2,3$):
     \[
     y_{i,t+1} \geq u_{i,t} \quad \forall i \in I, \; t=1,2,3
     \]
   - No startup allowed in period 4:
     \[
     u_{i,4} = 0 \quad \forall i \in I
     \]

3. **Minimum down-time (if shut down, must stay off for 2 periods):**
   - If shut down at $t$, must be off at $t+1$ and $t+2$ (for $t=1,2$):
     \[
     y_{i,t+1} \leq 1 - z_{i,t} \quad \forall i \in I, \; t=1,2
     \]
     \[
     y_{i,t+2} \leq 1 - z_{i,t} \quad \forall i \in I, \; t=1,2
     \]
   - If shut down at $t=3$, must be off at $t=4$:
     \[
     y_{i,4} \leq 1 - z_{i,3} \quad \forall i \in I
     \]

4. **Inactive trucks cannot transport goods:**
   \[
   x_{i,t} \leq Q_i y_{i,t} \quad \forall i \in I, \; t \in T
   \]
   \[
   x_{i,t} \geq 0 \quad \forall i \in I, \; t \in T
   \]

5. **Truck capacity:**
   \[
   x_{i,t} \leq Q_i y_{i,t} \quad \forall i \in I, \; t \in T
   \]

6. **Demand satisfaction:**
   \[
   \sum_{i \in I} x_{i,t} \geq d_t \quad \forall t \in T
   \]

7. **Spare capacity buffer (total load $\leq$ 90% of active capacity):**
   \[
   \sum_{i \in I} x_{i,t} \leq 0.9 \sum_{i \in I} Q_i y_{i,t} \quad \forall t \in T
   \]

8. **Load ramping constraint (change in load per truck per period $\leq 300$ kg):**
   - For $t=1$:
     \[
     |x_{i,1} - 0| \leq 300 \quad \forall i \in I
     \]
   - For $t=2,3,4$:
     \[
     |x_{i,t} - x_{i,t-1}| \leq 300 \quad \forall i \in I, \; t=2,3,4
     \]
   - These can be linearized as:
     \[
     x_{i,1} \leq 300
     \]
     \[
     -x_{i,1} \leq 300
     \]
     \[
     x_{i,t} - x_{i,t-1} \leq 300
     \]
     \[
     x_{i,t-1} - x_{i,t} \leq 300 \quad \forall i \in I, \; t=2,3,4
     \]

9. **Variable domains:**
   \[
   y_{i,t}, u_{i,t}, z_{i,t} \in \{0,1\} \quad \forall i \in I, \; t \in T
   \]
   \[
   x_{i,t} \geq 0 \quad \forall i \in I, \; t \in T
   \]

##### Parameters (full vectors):

- $Q = [1000, 800, 1200, 600, 900, 700, 1100, 500, 1000, 650]$
- $S = [500, 300, 400, 250, 450, 280, 420, 200, 480, 260]$
- $C = [2.0, 3.0, 2.5, 3.0, 2.2, 2.8, 2.4, 3.2, 2.1, 2.9]$
- $d = [1500, 2000, 1800, 1000]$

##### Summary

Minimize total startup and transportation costs by scheduling truck activations and transported weights, subject to startup/shutdown logic, minimum up/down times, ramping, capacity, demand, and buffer constraints, as detailed above.