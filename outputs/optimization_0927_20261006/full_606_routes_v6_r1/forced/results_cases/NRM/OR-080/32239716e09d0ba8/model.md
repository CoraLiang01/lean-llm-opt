#### Abstract Mathematical Model

**Index Sets:**
- $T$: Set of trucks (from column "truck_id" in file_0_view_0)
- $P$: Set of periods, $P = \{1,2,3,4\}$

**Parameters:**
- $Q_t$: Maximum capacity of truck $t$ (from column "Q", file_0_view_0)
- $S_t$: Startup cost for truck $t$ (from column "S", file_0_view_0)
- $C_t$: Unit transportation cost for truck $t$ (from column "C", file_0_view_0)
- $d_p$: Customer demand in period $p$ (from columns "d1", "d2", "d3", "d4", file_0_view_0; all trucks have the same demand per period)
- $M$: A sufficiently large constant (for logical constraints)

**Decision Variables:**
- $y_{t,p} \in \{0,1\}$: 1 if truck $t$ is active in period $p$, 0 otherwise
- $z_{t,p} \in \{0,1\}$: 1 if truck $t$ is started up at the beginning of period $p$, 0 otherwise
- $x_{t,p} \geq 0$: Amount of goods (kg) transported by truck $t$ in period $p$

**Objective:**
\[
\min \sum_{t \in T} \sum_{p \in P} \left( S_t \cdot z_{t,p} + C_t \cdot x_{t,p} \right)
\]

**Constraints:**

1. **Startup Logic:**
   - All trucks are initially off:
     \[
     y_{t,0} = 0 \quad \forall t \in T
     \]
   - Startup variable definition:
     \[
     z_{t,p} \geq y_{t,p} - y_{t,p-1} \quad \forall t \in T,\, p \in P
     \]
     \[
     z_{t,p} \leq 1 - y_{t,p-1} \quad \forall t \in T,\, p \in P
     \]
     \[
     z_{t,p} \leq y_{t,p} \quad \forall t \in T,\, p \in P
     \]
   - No startup allowed in period 4:
     \[
     z_{t,4} = 0 \quad \forall t \in T
     \]

2. **Minimum Up-Time (if started, must stay on at least 2 periods):**
   \[
   y_{t,p+1} \geq y_{t,p} - y_{t,p-1} \quad \forall t \in T,\, p = 1,2,3
   \]
   (If a truck is started in $p$, it must be on in $p+1$.)

3. **Minimum Down-Time (if shut down, must stay off for 2 periods):**
   \[
   y_{t,p+1} \leq y_{t,p} + y_{t,p-1} \quad \forall t \in T,\, p = 1,2,3
   \]
   (If a truck is off in $p+1$ and was on in $p$, it cannot be on again in $p+2$.)

4. **No restart before two periods after shutdown:**
   \[
   y_{t,p} + y_{t,p-1} \geq y_{t,p+1} \quad \forall t \in T,\, p = 1,2
   \]
   (If off in $p$ and $p-1$, cannot be on in $p+1$.)

5. **Inactive trucks transport zero:**
   \[
   x_{t,p} \leq Q_t \cdot y_{t,p} \quad \forall t \in T,\, p \in P
   \]

6. **Truck capacity:**
   \[
   x_{t,p} \leq Q_t \quad \forall t \in T,\, p \in P
   \]

7. **Load change limit (including transitions to/from zero):**
   \[
   |x_{t,p} - x_{t,p-1}| \leq 300 \quad \forall t \in T,\, p = 1,2,3,4
   \]
   (Define $x_{t,0} = 0$ for all $t$.)

   This can be linearized as:
   \[
   x_{t,p} - x_{t,p-1} \leq 300 \quad \forall t \in T,\, p \in P
   \]
   \[
   x_{t,p-1} - x_{t,p} \leq 300 \quad \forall t \in T,\, p \in P
   \]
   (Set $x_{t,0} = 0$.)

8. **Demand satisfaction:**
   \[
   \sum_{t \in T} x_{t,p} \geq d_p \quad \forall p \in P
   \]

9. **Spare-capacity buffer (total load $\leq$ 90% of active capacity):**
   \[
   \sum_{t \in T} x_{t,p} \leq 0.9 \sum_{t \in T} Q_t \cdot y_{t,p} \quad \forall p \in P
   \]

10. **Variable domains:**
    \[
    y_{t,p} \in \{0,1\},\quad z_{t,p} \in \{0,1\},\quad x_{t,p} \geq 0 \quad \forall t \in T,\, p \in P
    \]

**Data Mapping:**

- Table: parameters.csv (table_id: file_0_view_0)
    - Index set $T$: truck_id
    - Parameter $Q_t$: column "Q"
    - Parameter $S_t$: column "S"
    - Parameter $C_t$: column "C"
    - Parameter $d_p$: columns "d1", "d2", "d3", "d4" (identical for all trucks; use any row)
- All 10 trucks and all columns are included as returned by CSVQA.

**Notes:**
- All constraints and variable domains are unconditional as per the query.
- All trucks are initially off: $y_{t,0} = 0$ for all $t$.
- $x_{t,0} = 0$ for all $t$ (for load change constraints).
- No startup allowed in period 4: $z_{t,4} = 0$ for all $t$.
- The minimum up/down time and restart constraints are enforced as described.
- The model minimizes total startup and transportation costs as required.