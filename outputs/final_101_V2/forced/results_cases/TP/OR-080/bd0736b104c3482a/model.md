##### Sets and Indices

- $i \in \{1,2,\ldots,10\}$: truck index (see truck_id in source order)
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
- $d_t$: customer demand in period $t$ (kg): $d_1=1500$, $d_2=2000$, $d_3=1800$, $d_4=1000$

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

1. **Startup logic and initial state** (all trucks off before period 1):
   $$
   y_{i,1} = u_{i,1} \quad \forall i
   $$
   $$
   y_{i,t} - y_{i,t-1} \leq u_{i,t} \leq y_{i,t} \quad \forall i,\, t=2,3,4
   $$
   $$
   u_{i,4} = 0 \quad \forall i
   $$

2. **Minimum up-time (if started, must stay on at least 2 periods):**
   $$
   y_{i,t+1} \geq u_{i,t} \quad \forall i,\, t=1,2,3
   $$

3. **Minimum down-time (if shut down, must stay off at least 2 periods):**
   $$
   y_{i,t-1} - y_{i,t} \leq 1 - y_{i,t+1} \quad \forall i,\, t=2,3
   $$
   $$
   y_{i,t-1} - y_{i,t} \leq 1 - y_{i,t+2} \quad \forall i,\, t=2
   $$
   (Alternatively, for $t=2,3$, if $y_{i,t-1}=1$ and $y_{i,t}=0$, then $y_{i,t+1}=0$ and $y_{i,t+2}=0$.)

   More precisely, for $t=2$:
   $$
   y_{i,1} - y_{i,2} \leq 1 - y_{i,3}
   $$
   $$
   y_{i,1} - y_{i,2} \leq 1 - y_{i,4}
   $$
   For $t=3$:
   $$
   y_{i,2} - y_{i,3} \leq 1 - y_{i,4}
   $$

4. **Capacity and activity:**
   $$
   0 \leq x_{i,t} \leq Q_i y_{i,t} \quad \forall i,\, t
   $$

5. **Load change limit (including transitions to/from zero):**
   $$
   |x_{i,t} - x_{i,t-1}| \leq 300 \quad \forall i,\, t=2,3,4
   $$
   (Linearized as:)
   $$
   x_{i,t} - x_{i,t-1} \leq 300 \quad \forall i,\, t=2,3,4
   $$
   $$
   x_{i,t-1} - x_{i,t} \leq 300 \quad \forall i,\, t=2,3,4
   $$
   (Set $x_{i,0}=0$ for all $i$.)

6. **Demand satisfaction:**
   $$
   \sum_{i=1}^{10} x_{i,t} \geq d_t \quad \forall t=1,2,3,4
   $$

7. **Spare-capacity buffer (total load $\leq$ 90% of active capacity):**
   $$
   \sum_{i=1}^{10} x_{i,t} \leq 0.9 \sum_{i=1}^{10} Q_i y_{i,t} \quad \forall t=1,2,3,4
   $$

8. **Inactive trucks transport zero:**
   $$
   x_{i,t} \leq Q_i y_{i,t} \quad \forall i,\, t
   $$

9. **Variable domains:**
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

Demands: $d_1=1500$, $d_2=2000$, $d_3=1800$, $d_4=1000$

##### Notes

- All constraints and parameters are included as specified.
- All truck and period indices, parameters, and coefficients are preserved in source order.
- All variable domains and boundary semantics follow the user query.