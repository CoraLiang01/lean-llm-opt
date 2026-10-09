##### Sets and Indices

- $T = \{1,2,3,4\}$: Time periods (1 to 4)
- $K = \{1,2,\ldots,10\}$: Trucks (truck IDs 1 to 10)

##### Parameters

For each truck $k \in K$:
- $Q_k$: Maximum capacity of truck $k$
- $S_k$: Startup cost for truck $k$
- $C_k$: Unit transportation cost for truck $k$

Customer demand in period $t$:
- $d_t$: Demand in period $t$

From the data:

| $k$ | $Q_k$ | $S_k$ | $C_k$ |
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

For all $k \in K$, $t \in T$:
- $y_{k,t} \in \{0,1\}$: 1 if truck $k$ is active in period $t$, 0 otherwise
- $z_{k,t} \in \{0,1\}$: 1 if truck $k$ is started up at the beginning of period $t$, 0 otherwise
- $x_{k,t} \geq 0$: Amount of goods (kg) transported by truck $k$ in period $t$

##### Objective Function

Minimize total startup and transportation costs:
$$
\min \sum_{k=1}^{10} \sum_{t=1}^4 \left( S_k z_{k,t} + C_k x_{k,t} \right)
$$

##### Constraints

###### 1. Startup and Activation Logic

- All trucks are initially off:
  $$
  y_{k,0} = 0 \quad \forall k
  $$
  (Introduce $y_{k,0}$ as a fixed parameter for logic.)

- Startup definition:
  $$
  z_{k,t} \geq y_{k,t} - y_{k,t-1} \quad \forall k, \forall t \in T
  $$
  $$
  z_{k,t} \leq 1 - y_{k,t-1} \quad \forall k, \forall t \in T
  $$
  $$
  z_{k,t} \leq y_{k,t} \quad \forall k, \forall t \in T
  $$

- No startup allowed in period 4:
  $$
  z_{k,4} = 0 \quad \forall k
  $$

###### 2. Minimum Up-Time (Once started, must stay on at least 2 periods)

For all $k$, $t \in \{1,2,3\}$:
$$
y_{k,t+1} \geq z_{k,t} \quad \forall k, t=1,2,3
$$

###### 3. Minimum Down-Time (If shut down, must stay off for 2 periods)

For all $k$, $t \in \{1,2\}$:
$$
y_{k,t} - y_{k,t+1} \leq 1 - y_{k,t+1} - y_{k,t+2} \quad \forall k, t=1,2
$$
Alternatively, enforce:
$$
y_{k,t+1} + y_{k,t+2} \leq 1 + y_{k,t} \quad \forall k, t=1,2
$$
(If $y_{k,t}=1$ and $y_{k,t+1}=0$, then $y_{k,t+2}=0$.)

###### 4. Capacity and Activity

- Inactive trucks cannot transport goods:
  $$
  x_{k,t} \leq Q_k y_{k,t} \quad \forall k, t
  $$
  $$
  x_{k,t} \geq 0 \quad \forall k, t
  $$

###### 5. Ramping Constraints (Max 300 kg change per period)

For all $k$, $t=1$:
$$
|x_{k,1} - 0| \leq 300
$$
For all $k$, $t=2,3,4$:
$$
|x_{k,t} - x_{k,t-1}| \leq 300
$$
This can be linearized as:
$$
x_{k,t} - x_{k,t-1} \leq 300 \quad \forall k, t=2,3,4
$$
$$
x_{k,t-1} - x_{k,t} \leq 300 \quad \forall k, t=2,3,4
$$
And for $t=1$:
$$
x_{k,1} \leq 300
$$

###### 6. Demand Satisfaction

For all $t$:
$$
\sum_{k=1}^{10} x_{k,t} \geq d_t
$$

###### 7. Spare-Capacity Buffer (Total load $\leq$ 90% of active capacity)

For all $t$:
$$
\sum_{k=1}^{10} x_{k,t} \leq 0.9 \sum_{k=1}^{10} Q_k y_{k,t}
$$

###### 8. Variable Domains

- $y_{k,t} \in \{0,1\}$
- $z_{k,t} \in \{0,1\}$
- $x_{k,t} \geq 0$

##### Retrieved Information

{
  "trucks": [
    {"truck_id": 1, "Q": 1000, "S": 500, "C": 2.0},
    {"truck_id": 2, "Q": 800, "S": 300, "C": 3.0},
    {"truck_id": 3, "Q": 1200, "S": 400, "C": 2.5},
    {"truck_id": 4, "Q": 600, "S": 250, "C": 3.0},
    {"truck_id": 5, "Q": 900, "S": 450, "C": 2.2},
    {"truck_id": 6, "Q": 700, "S": 280, "C": 2.8},
    {"truck_id": 7, "Q": 1100, "S": 420, "C": 2.4},
    {"truck_id": 8, "Q": 500, "S": 200, "C": 3.2},
    {"truck_id": 9, "Q": 1000, "S": 480, "C": 2.1},
    {"truck_id": 10, "Q": 650, "S": 260, "C": 2.9}
  ],
  "demands": {
    "d1": 1500,
    "d2": 2000,
    "d3": 1800,
    "d4": 1000
  }
}

##### Summary

This model determines the activation, startup, and transported-weight schedule for 10 trucks over 4 periods, minimizing total startup and transportation costs, while satisfying all operational, ramping, demand, and buffer constraints as described.