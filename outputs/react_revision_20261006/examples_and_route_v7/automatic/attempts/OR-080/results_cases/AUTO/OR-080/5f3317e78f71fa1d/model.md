##### Decision Variables

Let $T = \{1,2,3,4\}$ (periods), $K = \{1,2,\ldots,10\}$ (trucks).

- $y_{k,t} \in \{0,1\}$: 1 if truck $k$ is active in period $t$, 0 otherwise.
- $z_{k,t} \in \{0,1\}$: 1 if truck $k$ is started up in period $t$, 0 otherwise.
- $x_{k,t} \geq 0$: weight (kg) transported by truck $k$ in period $t$ (continuous).

##### Parameters

From parameters.csv (table_id: file_0_view_0):

- $Q_k$: maximum capacity of truck $k$.
- $S_k$: startup cost of truck $k$.
- $C_k$: unit transportation cost of truck $k$.
- $d_t$: customer demand in period $t$.

##### Objective Function

Minimize total startup and transportation costs:
$$
\min \sum_{k\in K} \sum_{t\in T} \left( S_k z_{k,t} + C_k x_{k,t} \right)
$$

##### Constraints

1. **Startup logic and minimum up-time:**
   - Initial state: all trucks are off before period 1.
   - Startup definition:
     $$
     z_{k,1} \geq y_{k,1}
     $$
     $$
     z_{k,t} \geq y_{k,t} - y_{k,t-1} \quad \forall k\in K,\, t=2,3,4
     $$
   - Minimum up-time (if started, must stay on at least two periods; cannot start in period 4):
     $$
     z_{k,4} = 0 \quad \forall k\in K
     $$
     $$
     y_{k,t+1} \geq z_{k,t} \quad \forall k\in K,\, t=1,2,3
     $$

2. **Minimum down-time after shutdown:**
   - If a truck is shut down in period $t$ after being active in $t-1$, it must be off in $t$ and $t+1$:
     $$
     y_{k,t-1} - y_{k,t} \leq 1 - y_{k,t} - y_{k,t+1} \quad \forall k\in K,\, t=2,3
     $$
     (Alternatively, for $t=2,3$: if $y_{k,t-1}=1$ and $y_{k,t}=0$, then $y_{k,t+1}=0$.)

   - Cannot restart before $t+2$:
     $$
     y_{k,t-1} - y_{k,t} + y_{k,t+2} \leq 1 \quad \forall k\in K,\, t=2
     $$
     (If shut down at $t$, cannot be on at $t+2$.)

3. **Inactive trucks transport zero:**
   $$
   x_{k,t} \leq Q_k y_{k,t} \quad \forall k\in K,\, t\in T
   $$

4. **Truck capacity:**
   $$
   0 \leq x_{k,t} \leq Q_k y_{k,t} \quad \forall k\in K,\, t\in T
   $$

5. **Load ramping (change in transported weight per truck per period):**
   $$
   |x_{k,t} - x_{k,t-1}| \leq 300 \quad \forall k\in K,\, t=2,3,4
   $$
   (Introduce auxiliary variables or split into two inequalities:
   $$
   x_{k,t} - x_{k,t-1} \leq 300
   $$
   $$
   x_{k,t-1} - x_{k,t} \leq 300
   $$
   )

   For $x_{k,0}$, define $x_{k,0}=0$ (since all trucks are off before period 1).

6. **Demand satisfaction:**
   $$
   \sum_{k\in K} x_{k,t} \geq d_t \quad \forall t\in T
   $$

7. **Spare-capacity buffer (total load cannot exceed 90% of active trucks' combined capacity):**
   $$
   \sum_{k\in K} x_{k,t} \leq 0.9 \sum_{k\in K} Q_k y_{k,t} \quad \forall t\in T
   $$

8. **Variable domains:**
   $$
   y_{k,t} \in \{0,1\},\quad z_{k,t} \in \{0,1\},\quad x_{k,t} \geq 0
   $$

##### Data Mapping

- $K = \{$truck_id from file_0_view_0$\} = \{1,2,3,4,5,6,7,8,9,10\}$
- $T = \{1,2,3,4\}$
- $Q_k$ = value in column "Q" for truck $k$ (file_0_view_0)
- $S_k$ = value in column "S" for truck $k$ (file_0_view_0)
- $C_k$ = value in column "C" for truck $k$ (file_0_view_0)
- $d_t$ = value in column "d$t$" (file_0_view_0), for $t=1,2,3,4$

All indices, parameters, and constraints are mapped directly from the provided data and user description.