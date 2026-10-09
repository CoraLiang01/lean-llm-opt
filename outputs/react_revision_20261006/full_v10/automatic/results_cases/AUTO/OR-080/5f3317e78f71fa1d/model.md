## Mathematical Model

**Sets**
- $I$: set of trucks, $I = \{1,2,\ldots,10\}$ (from truck_id in parameters.csv)
- $T$: set of periods, $T = \{1,2,3,4\}$

**Parameters** (from parameters.csv, table_id: file_0_view_0)
- $Q_i$: maximum capacity of truck $i$ (kg), column Q
- $S_i$: startup cost for truck $i$ (yuan), column S
- $C_i$: unit transportation cost for truck $i$ (yuan/kg), column C
- $d_t$: customer demand in period $t$ (kg), columns d1, d2, d3, d4

**Decision Variables**
- $x_{i,t} \geq 0$: weight transported by truck $i$ in period $t$ (kg), continuous
- $y_{i,t} \in \{0,1\}$: 1 if truck $i$ is active in period $t$, 0 otherwise
- $z_{i,t} \in \{0,1\}$: 1 if truck $i$ is started up at the beginning of period $t$, 0 otherwise

**Objective**
Minimize total startup and transportation costs:
\[
\min \sum_{i \in I} \sum_{t \in T} S_i z_{i,t} + \sum_{i \in I} \sum_{t \in T} C_i x_{i,t}
\]

**Constraints**

1. **Demand satisfaction and spare capacity buffer**
   For all $t \in T$:
   \[
   \sum_{i \in I} x_{i,t} \geq d_t
   \]
   \[
   \sum_{i \in I} x_{i,t} \leq 0.9 \sum_{i \in I} Q_i y_{i,t}
   \]

2. **Truck capacity and activation**
   For all $i \in I$, $t \in T$:
   \[
   0 \leq x_{i,t} \leq Q_i y_{i,t}
   \]

3. **Startup logic**
   For all $i \in I$, $t=1$:
   \[
   z_{i,1} \geq y_{i,1}
   \]
   For all $i \in I$, $t=2,3,4$:
   \[
   z_{i,t} \geq y_{i,t} - y_{i,t-1}
   \]
   (A startup occurs if a truck is off in $t-1$ and on in $t$.)

4. **Minimum up-time (at least 2 consecutive periods)**
   For all $i \in I$, $t=1,2,3$:
   \[
   z_{i,t} \leq y_{i,t} + y_{i,t+1}
   \]
   (If started at $t$, must be on at $t$ and $t+1$.)

   For all $i \in I$:
   \[
   z_{i,4} = 0
   \]
   (No startup allowed in period 4.)

5. **Minimum down-time (at least 2 consecutive periods)**
   For all $i \in I$, $t=1,2$:
   \[
   y_{i,t} - y_{i,t+1} \leq 1 - y_{i,t+2}
   \]
   (If shut down at $t+1$, must be off at $t+2$.)

   For all $i \in I$, $t=3$:
   \[
   y_{i,3} - y_{i,4} \leq 1
   \]
   (If shut down at 4, no further periods.)

6. **Initial state**
   For all $i \in I$:
   \[
   y_{i,0} = 0
   \]
   (All trucks are initially off.)

7. **Load ramping**
   For all $i \in I$, $t=1$:
   \[
   |x_{i,1} - 0| \leq 300
   \]
   For all $i \in I$, $t=2,3,4$:
   \[
   |x_{i,t} - x_{i,t-1}| \leq 300
   \]

   (Can be linearized as:
   \[
   x_{i,t} - x_{i,t-1} \leq 300
   \]
   \[
   x_{i,t-1} - x_{i,t} \leq 300
   \]
   with $x_{i,0} = 0$.)

8. **Inactive trucks carry no load**
   For all $i \in I$, $t \in T$:
   \[
   x_{i,t} \leq Q_i y_{i,t}
   \]
   (Already included above.)

**Variable domains**
- $x_{i,t} \geq 0$ (continuous)
- $y_{i,t} \in \{0,1\}$
- $z_{i,t} \in \{0,1\}$

---

**Data Mapping**

- $I$ (trucks): truck_id in parameters.csv (file_0_view_0)
- $T$ (periods): $1,2,3,4$
- $Q_i$: column Q, file_0_view_0
- $S_i$: column S, file_0_view_0
- $C_i$: column C, file_0_view_0
- $d_t$: columns d1, d2, d3, d4, file_0_view_0 (identical for all trucks, use any row)

---

**Notes**
- All constraints and variables are indexed over the full set of trucks and periods as defined above.
- All logical and ramping constraints are fully linearizable as shown.
- All data is mapped directly from parameters.csv, table_id: file_0_view_0.