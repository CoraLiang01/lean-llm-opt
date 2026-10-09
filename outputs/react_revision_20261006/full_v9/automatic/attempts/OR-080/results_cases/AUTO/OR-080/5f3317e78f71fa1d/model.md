## Mathematical Model

### Sets
- $I$: set of trucks, $I = \{1,2,\ldots,10\}$ (from truck_id in parameters.csv)
- $T$: set of periods, $T = \{1,2,3,4\}$

### Parameters (from parameters.csv, table_id: file_0_view_0)
- $Q_i$: maximum capacity of truck $i$ (kg), column Q
- $S_i$: startup cost for truck $i$ (yuan), column S
- $C_i$: unit transportation cost for truck $i$ (yuan/kg), column C
- $d_t$: customer demand in period $t$ (kg), columns d1, d2, d3, d4

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

#### 1. Demand satisfaction and spare capacity buffer
For all $t \in T$:
\[
\sum_{i \in I} x_{i,t} \geq d_t
\]
\[
\sum_{i \in I} x_{i,t} \leq 0.9 \sum_{i \in I} Q_i y_{i,t}
\]

#### 2. Truck capacity and activity
For all $i \in I$, $t \in T$:
\[
0 \leq x_{i,t} \leq Q_i y_{i,t}
\]

#### 3. Startup logic
For all $i \in I$, $t \in T$:
\[
z_{i,t} \geq y_{i,t} - y_{i,t-1}
\]
where $y_{i,0} = 0$ (all trucks initially off).

#### 4. Minimum up-time (once started, must stay on at least 2 periods; cannot start in period 4)
For all $i \in I$:
\[
z_{i,4} = 0
\]
For all $i \in I$, $t \in \{1,2,3\}$:
\[
y_{i,t+1} \geq y_{i,t} - z_{i,t}
\]
(If started at $t$, must be on at $t+1$.)

#### 5. Minimum down-time (if shut down, must stay off for 2 periods)
For all $i \in I$, $t \in \{1,2\}$:
\[
y_{i,t} - y_{i,t+1} \leq 1 - y_{i,t+2}
\]
(If off at $t+1$ after being on at $t$, must be off at $t+2$.)

#### 6. Weight ramping (change in transported weight per truck per period $\leq 300$ kg)
For all $i \in I$, $t \in \{1,2,3\}$:
\[
|x_{i,t+1} - x_{i,t}| \leq 300
\]
This can be linearized as:
\[
x_{i,t+1} - x_{i,t} \leq 300
\]
\[
x_{i,t} - x_{i,t+1} \leq 300
\]

#### 7. Inactive trucks transport zero
For all $i \in I$, $t \in T$:
\[
x_{i,t} \leq Q_i y_{i,t}
\]

#### 8. Variable domains
\[
x_{i,t} \geq 0 \quad \forall i \in I, t \in T
\]
\[
y_{i,t} \in \{0,1\} \quad \forall i \in I, t \in T
\]
\[
z_{i,t} \in \{0,1\} \quad \forall i \in I, t \in T
\]

---

### Data Mapping

- $I$ (trucks): truck_id in parameters.csv, table_id: file_0_view_0
- $T$ (periods): $\{1,2,3,4\}$
- $Q_i$: column Q, table_id: file_0_view_0
- $S_i$: column S, table_id: file_0_view_0
- $C_i$: column C, table_id: file_0_view_0
- $d_1, d_2, d_3, d_4$: columns d1, d2, d3, d4, table_id: file_0_view_0

All constraints and variables are defined for the full set of trucks and periods as above.