##### Sets and Indices

- $i \in \{1,2,\ldots,10\}$: truck index (source order from parameters.csv)
- $t \in \{1,2,3,4\}$: time period

##### Parameters (from parameters.csv, source order)

| $i$ | $Q_i$ (max cap) | $S_i$ (startup cost) | $C_i$ (unit cost) |
|---|---|---|---|
| 1 | 1000 | 500 | 2.0 |
| 2 | 800  | 300 | 3.0 |
| 3 | 1200 | 400 | 2.5 |
| 4 | 600  | 250 | 3.0 |
| 5 | 900  | 450 | 2.2 |
| 6 | 700  | 280 | 2.8 |
| 7 | 1100 | 420 | 2.4 |
| 8 | 500  | 200 | 3.2 |
| 9 | 1000 | 480 | 2.1 |
| 10| 650  | 260 | 2.9 |

- Customer demand: $d_1=1500$, $d_2=2000$, $d_3=1800$, $d_4=1000$

##### Decision Variables

- $y_{i,t} \in \{0,1\}$: 1 if truck $i$ is active in period $t$, 0 otherwise
- $u_{i,t} \in \{0,1\}$: 1 if truck $i$ is started up at the start of period $t$, 0 otherwise
- $z_{i,t} \in \{0,1\}$: 1 if truck $i$ is shut down at the start of period $t$, 0 otherwise
- $x_{i,t} \geq 0$: weight transported by truck $i$ in period $t$ (kg), continuous

##### Objective

Minimize total startup and transportation costs:
$$
\min \sum_{i=1}^{10} \sum_{t=1}^4 S_i u_{i,t} + \sum_{i=1}^{10} \sum_{t=1}^4 C_i x_{i,t}
$$

##### Constraints

1. **Startup and shutdown logic** (initially off):
   $$
   y_{i,1} = u_{i,1} \qquad \forall i
   $$
   $$
   y_{i,t} - y_{i,t-1} = u_{i,t} - z_{i,t} \qquad \forall i,\, t=2,3,4
   $$
   $$
   y_{i,0} = 0 \qquad \forall i
   $$

2. **Minimum up-time (once started, must stay on at least 2 periods):**
   $$
   u_{i,t} \leq y_{i,t+1} \qquad \forall i,\, t=1,2,3
   $$
   $$
   u_{i,4} = 0 \qquad \forall i
   $$

3. **Minimum down-time (if shut down, must stay off for 2 periods):**
   $$
   z_{i,t} \leq 1 - y_{i,t+1} \qquad \forall i,\, t=1,2,3
   $$
   $$
   z_{i,t} \leq 1 - y_{i,t+2} \qquad \forall i,\, t=1,2
   $$
   $$
   z_{i,4} = 0 \qquad \forall i
   $$

4. **Inactive trucks transport zero:**
   $$
   x_{i,t} \leq Q_i y_{i,t} \qquad \forall i,\, t
   $$

5. **Truck capacity:**
   $$
   x_{i,t} \leq Q_i \qquad \forall i,\, t
   $$

6. **Demand satisfaction:**
   $$
   \sum_{i=1}^{10} x_{i,t} \geq d_t \qquad \forall t
   $$

7. **Spare capacity buffer (total load ≤ 90% of active capacity):**
   $$
   \sum_{i=1}^{10} x_{i,t} \leq 0.9 \sum_{i=1}^{10} Q_i y_{i,t} \qquad \forall t
   $$

8. **Load ramping (change ≤ 300 kg between adjacent periods, including to/from zero):**
   $$
   |x_{i,t} - x_{i,t-1}| \leq 300 \qquad \forall i,\, t=2,3,4
   $$
   $$
   x_{i,0} = 0 \qquad \forall i
   $$
   (Linearize with auxiliary variables if needed.)

9. **Non-negativity and integrality:**
   $$
   x_{i,t} \geq 0,\quad y_{i,t} \in \{0,1\},\quad u_{i,t} \in \{0,1\},\quad z_{i,t} \in \{0,1\}
   $$

##### Data (from parameters.csv, source order)

- $Q = [1000,\,800,\,1200,\,600,\,900,\,700,\,1100,\,500,\,1000,\,650]$
- $S = [500,\,300,\,400,\,250,\,450,\,280,\,420,\,200,\,480,\,260]$
- $C = [2.0,\,3.0,\,2.5,\,3.0,\,2.2,\,2.8,\,2.4,\,3.2,\,2.1,\,2.9]$
- $d_1=1500$, $d_2=2000$, $d_3=1800$, $d_4=1000$

##### Notes

- All indices, parameters, and constraints are in source order and use the exact values from parameters.csv.
- All constraints and variable domains are as specified in the user query.
- The model is a mixed-integer program (MIP) with binary and continuous variables.