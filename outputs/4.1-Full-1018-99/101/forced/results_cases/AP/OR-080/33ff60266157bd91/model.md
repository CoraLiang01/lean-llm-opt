##### Sets and Indices

- $i \in \{1,2,\ldots,10\}$: truck index (truck_id)
- $t \in \{1,2,3,4\}$: time period

##### Parameters (from parameters.csv)

- $Q_i$: maximum capacity of truck $i$
- $S_i$: startup cost for truck $i$
- $C_i$: unit transportation cost for truck $i$
- $d_t$: customer demand in period $t$

From the data:

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
- $z_{i,t} \in \{0,1\}$: 1 if truck $i$ is started up at the beginning of period $t$, 0 otherwise
- $x_{i,t} \geq 0$: weight transported by truck $i$ in period $t$ (kg)

##### Objective Function

Minimize total startup and transportation costs:
$$
\min \sum_{i=1}^{10} \sum_{t=1}^4 \left( S_i z_{i,t} + C_i x_{i,t} \right)
$$

##### Constraints

###### 1. Startup and Activation Logic

- Initial state: all trucks are off before period 1.
  $$
  y_{i,0} = 0 \quad \forall i
  $$
  (For modeling, define $y_{i,0} = 0$.)

- Startup definition:
  $$
  z_{i,t} \geq y_{i,t} - y_{i,t-1} \quad \forall i, \; t=1,\ldots,4
  $$
  $$
  z_{i,t} \leq 1 - y_{i,t-1} \quad \forall i, \; t=1,\ldots,4
  $$
  $$
  z_{i,t} \leq y_{i,t} \quad \forall i, \; t=1,\ldots,4
  $$

- No startup allowed in period 4:
  $$
  z_{i,4} = 0 \quad \forall i
  $$

###### 2. Minimum Up-Time (if started, must stay on at least 2 periods)

For $t=1,2,3$:
$$
y_{i,t+1} \geq z_{i,t} \quad \forall i, \; t=1,2,3
$$

###### 3. Minimum Down-Time (if shut down, must stay off for 2 periods)

For $t=1,2$:
$$
y_{i,t+1} \leq y_{i,t} \quad \forall i, \; t=1,2,3
$$
For $t=1,2$:
$$
y_{i,t+1} + y_{i,t+2} \leq 2 - (y_{i,t} - y_{i,t-1}) \quad \forall i, \; t=1,2
$$
Alternatively, more directly:
If $y_{i,t-1}=1$, $y_{i,t}=0$ (shutdown at $t$), then $y_{i,t+1}=0$, $y_{i,t+2}=0$.

So, for $t=1,2$:
$$
y_{i,t-1} - y_{i,t} \leq 1 - y_{i,t+1} \quad \forall i, \; t=1,2
$$
$$
y_{i,t-1} - y_{i,t} \leq 1 - y_{i,t+2} \quad \forall i, \; t=1,2
$$

###### 4. Capacity and Activity

- Inactive trucks transport zero:
  $$
  x_{i,t} \leq Q_i y_{i,t} \quad \forall i, t
  $$
  $$
  x_{i,t} \geq 0 \quad \forall i, t
  $$

###### 5. Load Change Limit

For $t=2,3,4$:
$$
x_{i,t} - x_{i,t-1} \leq 300 \quad \forall i
$$
$$
x_{i,t-1} - x_{i,t} \leq 300 \quad \forall i
$$
(Assume $x_{i,0} = 0$ for all $i$.)

###### 6. Demand Satisfaction

For each period:
$$
\sum_{i=1}^{10} x_{i,t} \geq d_t \quad \forall t=1,2,3,4
$$

###### 7. Spare Capacity Buffer

For each period:
$$
\sum_{i=1}^{10} x_{i,t} \leq 0.9 \sum_{i=1}^{10} Q_i y_{i,t} \quad \forall t=1,2,3,4
$$

###### 8. Variable Domains

$$
y_{i,t} \in \{0,1\} \quad \forall i, t
$$
$$
z_{i,t} \in \{0,1\} \quad \forall i, t
$$
$$
x_{i,t} \geq 0 \quad \forall i, t
$$

##### Retrieved Information

{
  "trucks": [
    {"truck_id": "1", "Q": 1000, "S": 500, "C": 2.0},
    {"truck_id": "2", "Q": 800, "S": 300, "C": 3.0},
    {"truck_id": "3", "Q": 1200, "S": 400, "C": 2.5},
    {"truck_id": "4", "Q": 600, "S": 250, "C": 3.0},
    {"truck_id": "5", "Q": 900, "S": 450, "C": 2.2},
    {"truck_id": "6", "Q": 700, "S": 280, "C": 2.8},
    {"truck_id": "7", "Q": 1100, "S": 420, "C": 2.4},
    {"truck_id": "8", "Q": 500, "S": 200, "C": 3.2},
    {"truck_id": "9", "Q": 1000, "S": 480, "C": 2.1},
    {"truck_id": "10", "Q": 650, "S": 260, "C": 2.9}
  ],
  "demands": {
    "d1": 1500,
    "d2": 2000,
    "d3": 1800,
    "d4": 1000
  }
}