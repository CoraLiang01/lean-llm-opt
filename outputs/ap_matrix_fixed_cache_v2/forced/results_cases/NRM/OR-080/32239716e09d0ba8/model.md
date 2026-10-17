#### Index Sets

- $T$: Set of time periods, $T = \{1,2,3,4\}$
- $K$: Set of trucks, as indexed by truck_id in parameters.csv

#### Parameters

- $Q_k$: Maximum capacity of truck $k$ (from column "Q", table_id: file_0_view_0)
- $S_k$: Startup cost for truck $k$ (from column "S", table_id: file_0_view_0)
- $C_k$: Unit transportation cost for truck $k$ (from column "C", table_id: file_0_view_0)
- $d_t$: Customer demand in period $t$ (from columns "d1", "d2", "d3", "d4", table_id: file_0_view_0)
- $\Delta$: Maximum allowed change in transported weight per truck between adjacent periods ($\Delta = 300$ kg)
- $\alpha$: Maximum allowed utilization of active fleet per period ($\alpha = 0.9$)

#### Decision Variables

- $y_{k,t} \in \{0,1\}$: 1 if truck $k$ is active in period $t$, 0 otherwise
- $z_{k,t} \in \{0,1\}$: 1 if truck $k$ is started up at the beginning of period $t$, 0 otherwise
- $x_{k,t} \geq 0$: Amount of goods (kg) transported by truck $k$ in period $t$

#### Objective

Minimize total startup and transportation costs:
$$
\min \sum_{k \in K} \sum_{t \in T} \left( S_k z_{k,t} + C_k x_{k,t} \right)
$$

#### Constraints

1. **Startup Logic and Initial State**
   - All trucks are initially off:
     $$
     y_{k,0} = 0 \quad \forall k \in K
     $$
   - Startup variable definition:
     $$
     z_{k,t} \geq y_{k,t} - y_{k,t-1} \quad \forall k \in K, \forall t \in T
     $$
     $$
     z_{k,t} \leq 1 - y_{k,t-1} \quad \forall k \in K, \forall t \in T
     $$
     $$
     z_{k,t} \leq y_{k,t} \quad \forall k \in K, \forall t \in T
     $$
   - No startups allowed in period 4:
     $$
     z_{k,4} = 0 \quad \forall k \in K
     $$

2. **Minimum Up-Time (if started, must stay on at least 2 periods)**
   - If started in period $t$ ($t \leq 3$), must be active in $t$ and $t+1$:
     $$
     y_{k,t+1} \geq z_{k,t} \quad \forall k \in K, \forall t = 1,2,3
     $$

3. **Minimum Down-Time (if shut down, must stay off for 2 periods)**
   - If truck is shut down in period $t$ after being active in $t-1$, must be off in $t$ and $t+1$:
     $$
     y_{k,t-1} - y_{k,t} \leq 1 - y_{k,t} - y_{k,t+1} \quad \forall k \in K, \forall t = 1,2,3
     $$
     (Alternatively, for $t=1,2,3$: if $y_{k,t-1}=1$ and $y_{k,t}=0$, then $y_{k,t+1}=0$.)

4. **Capacity and Activity Constraints**
   - Inactive trucks transport zero:
     $$
     x_{k,t} \leq Q_k y_{k,t} \quad \forall k \in K, \forall t \in T
     $$
     $$
     x_{k,t} \geq 0 \quad \forall k \in K, \forall t \in T
     $$

5. **Load Change Constraints**
   - For all $k \in K$, $t = 1,2,3$:
     $$
     |x_{k,t+1} - x_{k,t}| \leq \Delta
     $$
     (Can be linearized as two inequalities.)

6. **Demand Satisfaction**
   - For all $t \in T$:
     $$
     \sum_{k \in K} x_{k,t} \geq d_t
     $$

7. **Fleet Utilization Limit (10% buffer)**
   - For all $t \in T$:
     $$
     \sum_{k \in K} x_{k,t} \leq \alpha \sum_{k \in K} Q_k y_{k,t}
     $$

8. **Variable Domains**
   - $y_{k,t} \in \{0,1\}$, $z_{k,t} \in \{0,1\}$, $x_{k,t} \geq 0$ for all $k \in K$, $t \in T$

#### Data Mapping

- Table: parameters.csv (table_id: file_0_view_0)
    - Index set $K$: truck_id
    - Parameter $Q_k$: column "Q"
    - Parameter $S_k$: column "S"
    - Parameter $C_k$: column "C"
    - Parameters $d_1$, $d_2$, $d_3$, $d_4$: columns "d1", "d2", "d3", "d4"
- All 10 rows (trucks) are included, in source order.

#### Notes

- All constraints and variable domains are unconditional, as specified in the query.
- The model is abstract and does not include literal parameter values.
- The initial state $y_{k,0}=0$ is enforced for all trucks.
- The minimum up-time and down-time constraints are implemented as described in the query.
- The load change constraint is enforced for all adjacent periods, including transitions to/from zero.
- The 10% spare-capacity buffer is enforced via the $\alpha$ parameter.

This completes the abstract mathematical optimization model as requested.