[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal activation, startup, and transported-weight schedule for 10 candidate trucks over 4 periods to meet customer demand (with a 10% spare-capacity buffer), while minimizing total startup and transportation costs. The model must respect truck startup/shutdown rules, minimum up/down times, ramping (change in load) limits, and per-truck capacity constraints.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed-charge (startup) and ramping constraints.
3.  **Define Index Sets:** The primary indices are:
    - Trucks: \( i \in \{1, ..., 10\} \) (from 'truck_id')
    - Periods: \( t \in \{1, 2, 3, 4\} \)
4.  **Define Decision Variables:**
    -   `x[i, t]` = Amount of weight (kg) transported by truck \(i\) in period \(t\). Type: GRB.CONTINUOUS, \( x[i, t] \geq 0 \).
    -   `y[i, t]` = 1 if truck \(i\) is active (on) in period \(t\), 0 otherwise. Type: GRB.BINARY.
    -   `z[i, t]` = 1 if truck \(i\) is started up at the beginning of period \(t\), 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Truck maximum capacity: `Q` (per truck, from 'Q' column).
    -   Startup cost: `S` (per truck, from 'S' column).
    -   Unit transportation cost: `C` (per truck, from 'C' column).
    -   Customer demand per period: `d1`, `d2`, `d3`, `d4` (from columns 'd1'–'d4'; same for all trucks).
6.  **Formulate Objective:** Minimize total cost, which is the sum over all trucks and periods of:
    - Startup costs: \( S[i] \times z[i, t] \) (incurred when a truck is started in period \(t\))
    - Transportation costs: \( C[i] \times x[i, t] \)
    - Objective: Minimize \( \sum_{i=1}^{10} \sum_{t=1}^{4} (S[i] \cdot z[i, t] + C[i] \cdot x[i, t]) \)
7.  **Formulate Constraints:**
    -   **Demand Satisfaction:** For each period \(t\), total transported weight across all trucks must be at least the customer demand:
        - \( \sum_{i=1}^{10} x[i, t] \geq d_t \) (where \(d_t\) is from 'd1', 'd2', etc.)
    -   **Spare-Capacity Buffer:** For each period \(t\), total transported weight cannot exceed 90% of the combined maximum capacity of active trucks:
        - \( \sum_{i=1}^{10} x[i, t] \leq 0.9 \times \sum_{i=1}^{10} Q[i] \cdot y[i, t] \)
    -   **Truck Capacity:** For each truck and period, transported weight cannot exceed truck capacity if active, and must be zero if inactive:
        - \( x[i, t] \leq Q[i] \cdot y[i, t] \)
        - \( x[i, t] \geq 0 \)
    -   **Startup Variable Definition:** For each truck and period, startup occurs if truck is off in \(t-1\) and on in \(t\):
        - For \(t=1\): \( z[i, 1] = y[i, 1] \) (since all trucks are initially off)
        - For \(t>1\): \( z[i, t] \geq y[i, t] - y[i, t-1] \), and \( z[i, t] \leq y[i, t] \), \( z[i, t] \leq 1 - y[i, t-1] \)
    -   **Minimum Up-Time (Once Started, Stay On at Least 2 Periods):**
        - If \( z[i, t] = 1 \), then \( y[i, t+1] = 1 \) (for \( t = 1, 2, 3 \); cannot start in period 4)
        - Enforce \( z[i, 4] = 0 \) (no startup allowed in period 4)
    -   **Minimum Down-Time (If Shut Down, Stay Off at Least 2 Periods):**
        - If truck is on in \(t-1\) and off in \(t\) (i.e., \( y[i, t-1]=1, y[i, t]=0 \)), then \( y[i, t+1]=0 \) (for \( t=1,2 \)), and \( y[i, t+2]=0 \) (if within horizon)
        - More generally, after an on-to-off transition at \(t\), enforce \( y[i, t+1]=0 \) and \( y[i, t+2]=0 \) (if \(t+2\leq 4\))
    -   **No Startup in Period 4:** \( z[i, 4] = 0 \) for all \(i\)
    -   **Ramping (Change in Load) Constraints:** For each truck and adjacent periods, the absolute change in transported weight cannot exceed 300 kg:
        - For \( t=2,3,4 \): \( |x[i, t] - x[i, t-1]| \leq 300 \)
        - This includes transitions to/from zero (i.e., truck turning on/off)
    -   **Initial State:** All trucks are off before period 1: \( y[i, 0] = 0 \) (used for startup logic)
    -   **Inactive Truck Carries No Load:** \( x[i, t] = 0 \) whenever \( y[i, t] = 0 \)
[Abstract Model Plan END]