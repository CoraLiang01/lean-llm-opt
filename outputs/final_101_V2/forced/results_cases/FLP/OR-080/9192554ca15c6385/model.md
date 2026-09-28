##### Sets and Indices

- $I = \{1,2,\ldots,10\}$: set of trucks (truck_id)
- $T = \{1,2,3,4\}$: set of time periods

##### Parameters

- $Q_i$: Maximum capacity of truck $i$ (kg)
- $S_i$: Startup cost for truck $i$
- $C_i$: Unit transportation cost for truck $i$ (per kg)
- $d_t$: Customer demand in period $t$ (kg)

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

| $t$ | $d_t$ |
|-----|-------|
| 1   | 1500  |
| 2   | 2000  |
| 3   | 1800  |
| 4   | 1000  |

##### Decision Variables

- $x_{i,t} \geq 0$: Amount transported by truck $i$ in period $t$ (kg), continuous.
- $y_{i,t} \in \{0,1\}$: 1 if truck $i$ is active in period $t$, 0 otherwise.
- $z_{i,t} \in \{0,1\}$: 1 if truck $i$ is started up at the beginning of period $t$, 0 otherwise.

##### Objective Function

\[
\min \sum_{i \in I} \sum_{t \in T} S_i z_{i,t} + \sum_{i \in I} \sum_{t \in T} C_i x_{i,t}
\]

##### Constraints

1. **Demand satisfaction (minimum):**
   \[
   \sum_{i \in I} x_{i,t} \geq d_t \qquad \forall t \in T
   \]

2. **Spare capacity buffer (maximum):**
   \[
   \sum_{i \in I} x_{i,t} \leq 0.9 \sum_{i \in I} Q_i y_{i,t} \qquad \forall t \in T
   \]

3. **Truck capacity and activity:**
   \[
   0 \leq x_{i,t} \leq Q_i y_{i,t} \qquad \forall i \in I,\, t \in T
   \]

4. **Startup logic:**
   \[
   z_{i,1} \geq y_{i,1} \qquad \forall i \in I
   \]
   \[
   z_{i,t} \geq y_{i,t} - y_{i,t-1} \qquad \forall i \in I,\, t=2,3,4
   \]
   \[
   z_{i,4} = 0 \qquad \forall i \in I \quad \text{(no startup allowed in period 4)}
   \]
   \[
   y_{i,0} = 0 \qquad \forall i \in I \quad \text{(all trucks initially off)}
   \]

5. **Minimum up-time (if started, must stay on at least 2 periods):**
   \[
   y_{i,t+1} \geq z_{i,t} \qquad \forall i \in I,\, t=1,2,3
   \]
   (If started in $t$, must be on in $t+1$.)

6. **Minimum down-time (if shut down, must stay off for 2 periods):**
   \[
   y_{i,t} + y_{i,t+1} \leq 1 + y_{i,t-1} \qquad \forall i \in I,\, t=2,3
   \]
   (If $y_{i,t-1}=1$, $y_{i,t}=0$ implies $y_{i,t+1}=0$.)

   Alternatively, for all $i$ and $t=2,3$:
   \[
   y_{i,t-1} - y_{i,t} \leq 1 - y_{i,t+1}
   \]
   (If truck is shut down at $t$, must be off at $t+1$.)

   For $t=3$:
   \[
   y_{i,2} - y_{i,3} \leq 1 - y_{i,4}
   \]
   For $t=2$:
   \[
   y_{i,1} - y_{i,2} \leq 1 - y_{i,3}
   \]

7. **No restart before $t+2$ after shutdown:**
   \[
   y_{i,t-1} - y_{i,t} + y_{i,t+2} \leq 1 \qquad \forall i \in I,\, t=2
   \]
   (If shut down at $t$, cannot restart at $t+2$.)

8. **Weight change limit (ramp constraint):**
   \[
   |x_{i,t} - x_{i,t-1}| \leq 300 \qquad \forall i \in I,\, t=2,3,4
   \]
   (Introduce auxiliary variables or split into two inequalities:)
   \[
   x_{i,t} - x_{i,t-1} \leq 300
   \]
   \[
   x_{i,t-1} - x_{i,t} \leq 300
   \]
   For $x_{i,0} = 0$ (trucks initially off).

9. **Inactive trucks transport zero:**
   \[
   x_{i,t} \leq Q_i y_{i,t} \qquad \forall i \in I,\, t \in T
   \]
   (Already included above.)

##### Variable Domains

- $x_{i,t} \geq 0$ (continuous)
- $y_{i,t} \in \{0,1\}$
- $z_{i,t} \in \{0,1\}$

##### Parameters (explicit values)

- $I = \{1,2,3,4,5,6,7,8,9,10\}$
- $T = \{1,2,3,4\}$
- $Q = [1000, 800, 1200, 600, 900, 700, 1100, 500, 1000, 650]$
- $S = [500, 300, 400, 250, 450, 280, 420, 200, 480, 260]$
- $C = [2.0, 3.0, 2.5, 3.0, 2.2, 2.8, 2.4, 3.2, 2.1, 2.9]$
- $d = [1500, 2000, 1800, 1000]$

##### Summary of Model

Minimize total startup and transportation costs, subject to:

- Demand satisfaction and spare capacity buffer in each period
- Truck capacity and activity logic
- Startup and minimum up/down time constraints
- Weight ramping constraints
- Inactive trucks transport zero

All parameters and constraints are explicitly included as per the CSV data and problem description.