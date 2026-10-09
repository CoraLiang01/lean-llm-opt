## Mathematical Model

**Sets**
- $I$: set of trucks, $I = \{1,2,\ldots,10\}$ (from truck_id in parameters.csv)
- $T$: set of periods, $T = \{1,2,3,4\}$

**Parameters** (from parameters.csv, table_id: file_0_view_0)
- $Q_i$: maximum capacity of truck $i$ (kg)
- $S_i$: startup cost for truck $i$
- $C_i$: unit transportation cost for truck $i$
- $d_t$: customer demand in period $t$ (kg)
- $M$: a large constant (e.g., $\max_i Q_i$)

**Variables**
- $x_{i,t} \geq 0$: weight transported by truck $i$ in period $t$ (kg)
- $y_{i,t} \in \{0,1\}$: 1 if truck $i$ is active in period $t$, 0 otherwise
- $z_{i,t} \in \{0,1\}$: 1 if truck $i$ is started up at the beginning of period $t$, 0 otherwise

**Objective**
\[
\min \sum_{i \in I} \sum_{t \in T} \left( S_i z_{i,t} + C_i x_{i,t} \right)
\]

**Constraints**

1. **Demand satisfaction**
   \[
   \sum_{i \in I} x_{i,t} \geq d_t \qquad \forall t \in T
   \]

2. **Spare-capacity buffer (90% of active capacity)**
   \[
   \sum_{i \in I} x_{i,t} \leq 0.9 \sum_{i \in I} Q_i y_{i,t} \qquad \forall t \in T
   \]

3. **Truck capacity and activity**
   \[
   0 \leq x_{i,t} \leq Q_i y_{i,t} \qquad \forall i \in I,\, t \in T
   \]

4. **Startup logic**
   \[
   z_{i,1} \geq y_{i,1} \qquad \forall i \in I
   \]
   \[
   z_{i,t} \geq y_{i,t} - y_{i,t-1} \qquad \forall i \in I,\, t=2,3,4
   \]
   \[
   z_{i,4} = 0 \qquad \forall i \in I
   \]
   (No startup allowed in period 4)

5. **Initial state**
   \[
   y_{i,0} = 0 \qquad \forall i \in I
   \]
   (All trucks off before period 1)

6. **Minimum up-time (if started, must stay on at least 2 periods)**
   \[
   y_{i,t+1} \geq y_{i,t} - z_{i,t} \qquad \forall i \in I,\, t=1,2,3
   \]
   (If started at $t$, must be on at $t+1$)

7. **No startup in period 4**
   \[
   z_{i,4} = 0 \qquad \forall i \in I
   \]

8. **Minimum down-time (if shut down, must stay off for 2 periods)**
   \[
   y_{i,t} + y_{i,t+1} \leq 1 + y_{i,t-1} \qquad \forall i \in I,\, t=2,3
   \]
   (If $y_{i,t-1}=1$ and $y_{i,t}=0$, then $y_{i,t+1}=0$)

   Alternatively, for $t=2,3$:
   \[
   y_{i,t-1} - y_{i,t} \leq 1 - y_{i,t+1} \qquad \forall i \in I,\, t=2,3
   \]
   (If off at $t$ after being on at $t-1$, must be off at $t+1$)

   For $t=3$:
   \[
   y_{i,2} - y_{i,3} \leq 1 - y_{i,4}
   \]
   For $t=2$:
   \[
   y_{i,1} - y_{i,2} \leq 1 - y_{i,3}
   \]

9. **No restart before two periods after shutdown**
   \[
   y_{i,t-1} - y_{i,t} \leq 1 - y_{i,t+1} \qquad \forall i \in I,\, t=2,3
   \]
   (As above)

10. **No startup in period 4**
    (Already enforced above)

11. **Load ramping constraint**
    \[
    |x_{i,t} - x_{i,t-1}| \leq 300 \qquad \forall i \in I,\, t=2,3,4
    \]
    (Set $x_{i,0}=0$ for all $i$)

    Linearized as:
    \[
    x_{i,t} - x_{i,t-1} \leq 300
    \]
    \[
    x_{i,t-1} - x_{i,t} \leq 300
    \]
    for $t=2,3,4$, $x_{i,0}=0$

12. **Inactive truck must transport zero**
    \[
    x_{i,t} \leq Q_i y_{i,t} \qquad \forall i \in I,\, t \in T
    \]
    (Already included above)

**Variable domains**
- $x_{i,t} \geq 0$ (continuous)
- $y_{i,t} \in \{0,1\}$
- $z_{i,t} \in \{0,1\}$

---

### Data Mapping

- $I$ (truck index): truck_id in parameters.csv (table_id: file_0_view_0)
- $T$ (periods): $t=1,2,3,4$
- $Q_i$: column Q, table_id: file_0_view_0
- $S_i$: column S, table_id: file_0_view_0
- $C_i$: column C, table_id: file_0_view_0
- $d_t$: columns d1, d2, d3, d4, table_id: file_0_view_0 (identical for all trucks)
- All constraints and variables as above

---

**Note:** All indices, parameters, and constraints are mapped directly to the columns and rows of parameters.csv (table_id: file_0_view_0) as described.