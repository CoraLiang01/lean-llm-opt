##### Sets and Indices

- $i \in \{1,2,\ldots,10\}$: truck index (truck_id from parameters.csv)
- $t \in \{1,2,3,4\}$: time period

##### Parameters (from parameters.csv, source order)

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

- $Q_i$: maximum capacity of truck $i$ (kg)
- $S_i$: startup cost for truck $i$
- $C_i$: unit transportation cost for truck $i$ (per kg)
- Customer demand: $d_1=1500$, $d_2=2000$, $d_3=1800$, $d_4=1000$

##### Decision Variables

- $y_{i,t} \in \{0,1\}$: 1 if truck $i$ is active in period $t$, 0 otherwise
- $u_{i,t} \in \{0,1\}$: 1 if truck $i$ is started up at the beginning of period $t$, 0 otherwise
- $x_{i,t} \geq 0$: weight transported by truck $i$ in period $t$ (kg)

##### Objective

Minimize total startup and transportation costs:
$$
\min \sum_{i=1}^{10} \sum_{t=1}^4 S_i u_{i,t} + \sum_{i=1}^{10} \sum_{t=1}^4 C_i x_{i,t}
$$

##### Constraints

1. **Startup logic and initial state**  
   All trucks are initially off:
   $$
   y_{i,0} = 0 \quad \forall i
   $$
   Startup variable definition:
   $$
   u_{i,t} \geq y_{i,t} - y_{i,t-1} \quad \forall i,\, t=1,\ldots,4
   $$
   $$
   u_{i,t} \leq 1 \quad \forall i,\, t=1,\ldots,4
   $$
   $$
   u_{i,4} = 0 \quad \forall i
   $$

2. **Minimum up-time (once started, must stay on at least 2 periods; cannot start in period 4):**
   $$
   y_{i,t+1} \geq y_{i,t} - u_{i,t} \quad \forall i,\, t=1,2,3
   $$
   $$
   y_{i,4} \geq y_{i,3} - u_{i,3} \quad \forall i
   $$
   $$
   u_{i,4} = 0 \quad \forall i
   $$

   Alternatively, enforce:  
   If $u_{i,t}=1$ for $t=1,2,3$, then $y_{i,t+1}=1$.

3. **Minimum down-time (if shut down, must stay off for 2 periods):**
   For $t=2,3$:
   $$
   y_{i,t-1} - y_{i,t} \leq 1 - y_{i,t+1} \quad \forall i,\, t=2,3
   $$
   For $t=3$:
   $$
   y_{i,2} - y_{i,3} \leq 1 - y_{i,4} \quad \forall i
   $$
   For $t=4$ (cannot restart in $t=4$):
   $$
   y_{i,3} - y_{i,4} \leq 1 \quad \forall i
   $$

   Alternatively, for all $t=2,3$:
   $$
   y_{i,t-1} - y_{i,t} \leq 1 - y_{i,t+1}
   $$
   and $y_{i,4}$ cannot be started up.

4. **Link transported weight to activation:**
   $$
   0 \leq x_{i,t} \leq Q_i y_{i,t} \quad \forall i,\, t=1,\ldots,4
   $$

5. **Demand satisfaction:**
   $$
   \sum_{i=1}^{10} x_{i,t} \geq d_t \quad \forall t=1,\ldots,4
   $$

6. **Spare-capacity buffer (total transported weight $\leq$ 90% of active capacity):**
   $$
   \sum_{i=1}^{10} x_{i,t} \leq 0.9 \sum_{i=1}^{10} Q_i y_{i,t} \quad \forall t=1,\ldots,4
   $$

7. **Weight ramping (change in transported weight per truck per period $\leq 300$ kg):**
   $$
   |x_{i,t} - x_{i,t-1}| \leq 300 \quad \forall i,\, t=1,\ldots,4
   $$
   where $x_{i,0} = 0$ for all $i$.

   This can be linearized as:
   $$
   x_{i,t} - x_{i,t-1} \leq 300 \quad \forall i,\, t=1,\ldots,4
   $$
   $$
   x_{i,t-1} - x_{i,t} \leq 300 \quad \forall i,\, t=1,\ldots,4
   $$
   with $x_{i,0}=0$.

8. **Variable domains:**
   $$
   y_{i,t} \in \{0,1\},\quad u_{i,t} \in \{0,1\},\quad x_{i,t} \geq 0
   $$

##### Data (from parameters.csv, source order)

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

Customer demands: $d_1=1500$, $d_2=2000$, $d_3=1800$, $d_4=1000$

##### Summary of Variables

- $y_{i,t}$: truck $i$ active in period $t$ (binary)
- $u_{i,t}$: truck $i$ started up at period $t$ (binary)
- $x_{i,t}$: weight transported by truck $i$ in period $t$ (continuous, $\geq 0$)

All constraints and coefficients are preserved as in the data and question.