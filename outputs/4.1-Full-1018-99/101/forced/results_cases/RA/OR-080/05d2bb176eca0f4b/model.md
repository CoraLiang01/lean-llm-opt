Let:
- $T = \{1,2,3,4\}$ (time periods)
- $K = \{1,2,3,4,5,6,7,8,9,10\}$ (truck_id, as in parameters.csv)
- $Q_k$ = maximum capacity of truck $k$
- $S_k$ = startup cost of truck $k$
- $C_k$ = unit transportation cost of truck $k$
- $d_t$ = customer demand in period $t$

Parameters (from parameters.csv, source order):

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
- $u_{k,t} \in \{0,1\}$: 1 if truck $k$ is started up at the beginning of period $t$, 0 otherwise
- $x_{k,t} \geq 0$: weight transported by truck $k$ in period $t$ (kg)

Model:

Minimize total cost:
$$
\min \sum_{k \in K} \sum_{t \in T} \left( S_k u_{k,t} + C_k x_{k,t} \right)
$$

Subject to:

1. **Startup logic and initial state** (all trucks off before period 1):
   $$
   y_{k,1} = u_{k,1} \quad \forall k \in K
   $$
   $$
   y_{k,t} - y_{k,t-1} = u_{k,t} - v_{k,t} \quad \forall k \in K, \ t=2,3,4
   $$
   Where $v_{k,t}$ is a shutdown variable (not strictly needed for this model, see below).

   Alternatively, since only startups are costed and shutdowns are only relevant for minimum up/down time, we can define:
   $$
   u_{k,1} = y_{k,1} \quad \forall k
   $$
   $$
   u_{k,t} \geq y_{k,t} - y_{k,t-1} \quad \forall k, \ t=2,3,4
   $$
   $$
   u_{k,t} \leq 1 - y_{k,t-1} \quad \forall k, \ t=2,3,4
   $$
   $$
   u_{k,t} \leq y_{k,t} \quad \forall k, \ t=2,3,4
   $$

2. **No startups in period 4:**
   $$
   u_{k,4} = 0 \quad \forall k \in K
   $$

3. **Minimum up time (once started, must stay on at least 2 consecutive periods):**
   $$
   y_{k,t+1} \geq u_{k,t} \quad \forall k \in K, \ t=1,2,3
   $$
   (If started in $t$, must be on in $t+1$.)

4. **Minimum down time (if shut down, must stay off for 2 periods):**
   $$
   y_{k,t-1} - y_{k,t} \leq 1 - y_{k,t+1} \quad \forall k \in K, \ t=2,3
   $$
   $$
   y_{k,t-1} - y_{k,t} \leq 1 - y_{k,t+2} \quad \forall k \in K, \ t=2
   $$
   (If $y_{k,t-1}=1$, $y_{k,t}=0$, then $y_{k,t+1}=0$, $y_{k,t+2}=0$.)

   Alternatively, for all $k$ and $t=2,3$:
   $$
   y_{k,t-1} - y_{k,t} \leq 1 - y_{k,t+1}
   $$
   For $t=2$:
   $$
   y_{k,1} - y_{k,2} \leq 1 - y_{k,3}
   $$
   $$
   y_{k,1} - y_{k,2} \leq 1 - y_{k,4}
   $$
   For $t=3$:
   $$
   y_{k,2} - y_{k,3} \leq 1 - y_{k,4}
   $$

5. **Inactive trucks transport zero:**
   $$
   x_{k,t} \leq Q_k y_{k,t} \quad \forall k \in K, \ t \in T
   $$

6. **Truck capacity:**
   $$
   x_{k,t} \leq Q_k \quad \forall k \in K, \ t \in T
   $$

7. **Load change limit (including transitions to/from zero):**
   $$
   |x_{k,t} - x_{k,t-1}| \leq 300 \quad \forall k \in K, \ t=2,3,4
   $$
   (Can be written as two inequalities:
   $$
   x_{k,t} - x_{k,t-1} \leq 300
   $$
   $$
   x_{k,t-1} - x_{k,t} \leq 300
   $$
   with $x_{k,0} = 0$ for all $k$.)

8. **Demand satisfaction:**
   $$
   \sum_{k \in K} x_{k,t} \geq d_t \quad \forall t \in T
   $$

9. **Spare capacity buffer (total load $\leq$ 90% of active capacity):**
   $$
   \sum_{k \in K} x_{k,t} \leq 0.9 \sum_{k \in K} Q_k y_{k,t} \quad \forall t \in T
   $$

10. **Variable domains:**
   $$
   y_{k,t} \in \{0,1\} \quad \forall k \in K, \ t \in T
   $$
   $$
   u_{k,t} \in \{0,1\} \quad \forall k \in K, \ t \in T
   $$
   $$
   x_{k,t} \geq 0 \quad \forall k \in K, \ t \in T
   $$

**Parameter values (from parameters.csv, source order):**

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

- $d_1=1500$, $d_2=2000$, $d_3=1800$, $d_4=1000$

**All indices, parameters, and constraints use the original truck_id and period order as in the data.**