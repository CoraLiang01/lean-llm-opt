## Mathematical Model

### Sets
- $I$: set of trucks, $I = \{1,2,\ldots,10\}$ (from truck_id in parameters.csv)
- $T$: set of periods, $T = \{1,2,3,4\}$

### Parameters (from parameters.csv, table_id: file_0_view_0)
- $Q_i$: maximum capacity of truck $i$ (kg)
- $S_i$: startup cost for truck $i$
- $C_i$: unit transportation cost for truck $i$
- $d_t$: customer demand in period $t$ (kg)
    - $d_1 = 1500$, $d_2 = 2000$, $d_3 = 1800$, $d_4 = 1000$
- $M$: a sufficiently large constant (e.g., $M \geq \max_i Q_i$)

### Decision Variables
- $x_{i,t} \geq 0$: weight transported by truck $i$ in period $t$ (kg), continuous
- $y_{i,t} \in \{0,1\}$: 1 if truck $i$ is active in period $t$, 0 otherwise
- $z_{i,t} \in \{0,1\}$: 1 if truck $i$ is started up at the beginning of period $t$, 0 otherwise

### Objective
Minimize total cost:
\[
\min \sum_{i \in I} \sum_{t \in T} \left( S_i z_{i,t} + C_i x_{i,t} \right)
\]

### Constraints

#### 1. Demand satisfaction (file_0_view_0, d1-d4)
\[
\sum_{i \in I} x_{i,t} \geq d_t \qquad \forall t \in T
\]

#### 2. Spare-capacity buffer (file_0_view_0, Q)
\[
\sum_{i \in I} x_{i,t} \leq 0.9 \sum_{i \in I} Q_i y_{i,t} \qquad \forall t \in T
\]

#### 3. Truck capacity and activity
\[
0 \leq x_{i,t} \leq Q_i y_{i,t} \qquad \forall i \in I,\, t \in T
\]

#### 4. Startup logic (minimum up-time, initial state, no startup in period 4)
- All trucks are initially off:
    \[
    y_{i,0} = 0 \qquad \forall i \in I
    \]
- Startup variable definition:
    \[
    z_{i,t} \geq y_{i,t} - y_{i,t-1} \qquad \forall i \in I,\, t \in T
    \]
    (with $y_{i,0} = 0$)
- No startup in period 4:
    \[
    z_{i,4} = 0 \qquad \forall i \in I
    \]
- Minimum up-time (if started, must stay on at least two periods):
    \[
    y_{i,t+1} \geq z_{i,t} \qquad \forall i \in I,\, t = 1,2,3
    \]
    (no startup in $t=4$)

#### 5. Minimum down-time (if shut down, must stay off for two periods)
- Define shutdowns:
    \[
    s_{i,t} \geq y_{i,t-1} - y_{i,t} \qquad \forall i \in I,\, t = 1,2,3,4
    \]
    (with $y_{i,0} = 0$)
- If shut down at $t$, must be off at $t+1$ and $t+2$:
    \[
    y_{i,t+1} \leq 1 - s_{i,t} \qquad \forall i \in I,\, t = 1,2,3
    \]
    \[
    y_{i,t+2} \leq 1 - s_{i,t} \qquad \forall i \in I,\, t = 1,2
    \]
    (for $t+1 \leq 4$, $t+2 \leq 4$; ignore constraints for $t+1>4$ or $t+2>4$)

#### 6. Load ramping (file_0_view_0, Q)
\[
|x_{i,t} - x_{i,t-1}| \leq 300 \qquad \forall i \in I,\, t = 1,2,3,4
\]
(with $x_{i,0} = 0$)

#### 7. Inactive trucks cannot transport
\[
x_{i,t} \leq Q_i y_{i,t} \qquad \forall i \in I,\, t \in T
\]

#### 8. Variable domains
\[
x_{i,t} \geq 0 \qquad \forall i \in I,\, t \in T
\]
\[
y_{i,t} \in \{0,1\} \qquad \forall i \in I,\, t \in T
\]
\[
z_{i,t} \in \{0,1\} \qquad \forall i \in I,\, t \in T
\]
\[
s_{i,t} \in \{0,1\} \qquad \forall i \in I,\, t = 1,2,3,4
\]

### Data Mapping

- $I$ (trucks): truck_id in parameters.csv (file_0_view_0)
- $T$ (periods): $\{1,2,3,4\}$
- $Q_i$: column Q, file_0_view_0
- $S_i$: column S, file_0_view_0
- $C_i$: column C, file_0_view_0
- $d_t$: columns d1, d2, d3, d4, file_0_view_0 (same for all trucks)
- All constraints and variables are indexed as above.

**Note:** $y_{i,0} = 0$, $x_{i,0} = 0$ for all $i$ (initial state). $s_{i,t}$ is an auxiliary binary variable for shutdowns.

---

This model fully encodes the described requirements, using only the current data and explicit query logic.