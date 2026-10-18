##### Sets and Indices

- $I = \{1,2,\ldots,10\}$: set of trucks, indexed by $i$
- $T = \{1,2,3,4\}$: set of time periods, indexed by $t$

##### Parameters

From parameters.csv:

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

Customer demands:
- $d_1 = 1500$
- $d_2 = 2000$
- $d_3 = 1800$
- $d_4 = 1000$

##### Decision Variables

- $x_{i,t} \geq 0$: weight transported by truck $i$ in period $t$ (continuous)
- $y_{i,t} \in \{0,1\}$: 1 if truck $i$ is active in period $t$, 0 otherwise
- $u_{i,t} \in \{0,1\}$: 1 if truck $i$ is started up at the beginning of period $t$, 0 otherwise
- $z_{i,t} \in \{0,1\}$: 1 if truck $i$ is shut down at the end of period $t$, 0 otherwise

##### Objective Function

\[
\min \sum_{i \in I} \sum_{t \in T} \left( S_i u_{i,t} + C_i x_{i,t} \right)
\]

##### Constraints

1. **Startup and Activation Logic**

   - Initial state: all trucks are off before period 1.
     \[
     y_{i,0} = 0 \quad \forall i \in I
     \]
     (Introduce $y_{i,0}$ as a fixed parameter.)

   - Startup definition:
     \[
     u_{i,t} \geq y_{i,t} - y_{i,t-1} \quad \forall i \in I, \; t \in T
     \]
     \[
     u_{i,t} \leq 1 - y_{i,t-1} \quad \forall i \in I, \; t \in T
     \]
     \[
     u_{i,t} \leq y_{i,t} \quad \forall i \in I, \; t \in T
     \]
     (Ensures $u_{i,t}=1$ iff truck $i$ is started up in $t$.)

   - No startup allowed in period 4:
     \[
     u_{i,4} = 0 \quad \forall i \in I
     \]

   - Shutdown definition:
     \[
     z_{i,t} \geq y_{i,t} - y_{i,t+1} \quad \forall i \in I, \; t = 1,2,3
     \]
     \[
     z_{i,t} \leq y_{i,t} \quad \forall i \in I, \; t = 1,2,3
     \]
     \[
     z_{i,t} \leq 1 - y_{i,t+1} \quad \forall i \in I, \; t = 1,2,3
     \]
     (Set $y_{i,5}=0$ for $t=4$.)

2. **Minimum Up-Time (Once started, must stay on at least 2 periods)**

   For all $i \in I$, $t = 1,2,3$:
   \[
   y_{i,t+1} \geq u_{i,t}
   \]
   (If started at $t$, must be on at $t+1$.)

3. **Minimum Down-Time (If shut down, must stay off for at least 2 periods)**

   For all $i \in I$, $t = 1,2$:
   \[
   y_{i,t+1} \leq 1 - z_{i,t}
   \]
   \[
   y_{i,t+2} \leq 1 - z_{i,t}
   \]
   (If shut down at $t$, must be off at $t+1$ and $t+2$.)

4. **Capacity and Activity Coupling**

   - Inactive trucks cannot transport goods:
     \[
     x_{i,t} \leq Q_i y_{i,t} \quad \forall i \in I, \; t \in T
     \]
     \[
     x_{i,t} \geq 0 \quad \forall i \in I, \; t \in T
     \]

5. **Demand Satisfaction and Buffer**

   - Total transported weight at least demand:
     \[
     \sum_{i \in I} x_{i,t} \geq d_t \quad \forall t \in T
     \]
   - Total transported weight at most 90% of active capacity:
     \[
     \sum_{i \in I} x_{i,t} \leq 0.9 \sum_{i \in I} Q_i y_{i,t} \quad \forall t \in T
     \]

6. **Load Change Limits**

   For all $i \in I$, $t = 2,3,4$:
   \[
   |x_{i,t} - x_{i,t-1}| \leq 300
   \]
   This can be linearized as:
   \[
   x_{i,t} - x_{i,t-1} \leq 300
   \]
   \[
   x_{i,t-1} - x_{i,t} \leq 300
   \]
   For $t=1$, set $x_{i,0}=0$.

7. **Initial Conditions**

   \[
   y_{i,0} = 0, \quad x_{i,0} = 0 \quad \forall i \in I
   \]

##### Variable Domains

- $x_{i,t} \geq 0$ (continuous)
- $y_{i,t} \in \{0,1\}$
- $u_{i,t} \in \{0,1\}$
- $z_{i,t} \in \{0,1\}$

---

###### Retrieved Parameters

- $I = \{1,2,3,4,5,6,7,8,9,10\}$
- $T = \{1,2,3,4\}$
- $Q = [1000, 800, 1200, 600, 900, 700, 1100, 500, 1000, 650]$
- $S = [500, 300, 400, 250, 450, 280, 420, 200, 480, 260]$
- $C = [2.0, 3.0, 2.5, 3.0, 2.2, 2.8, 2.4, 3.2, 2.1, 2.9]$
- $d = [1500, 2000, 1800, 1000]$

---

**All constraints and parameters are included as described in the problem and CSV.**