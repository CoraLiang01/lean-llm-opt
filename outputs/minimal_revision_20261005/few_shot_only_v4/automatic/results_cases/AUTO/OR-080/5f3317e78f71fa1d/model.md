[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal activation, startup, and transported-weight schedule for 10 candidate trucks over 4 consecutive periods to meet customer demand (with a 10% spare-capacity buffer), while minimizing total startup and transportation costs. The model must respect truck startup/shutdown rules, minimum up/down times, ramping (change in load) limits, and per-truck capacity constraints.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed-charge (startup) and ramping constraints.
3.  **Define Index Sets:** The primary indices are:
    - Trucks: \( i \in \{1,2,\ldots,10\} \) (from 'truck_id' in parameters.csv)
    - Periods: \( t \in \{1,2,3,4\} \)
4.  **Define Decision Variables:**
    -   `x[i,t]` = Amount of goods (kg) transported by truck \( i \) in period \( t \). Type: GRB.CONTINUOUS, \( x[i,t] \geq 0 \).
    -   `y[i,t]` = 1 if truck \( i \) is active (on) in period \( t \), 0 otherwise. Type: GRB.BINARY.
    -   `u[i,t]` = 1 if truck \( i \) is started up at the beginning of period \( t \), 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Truck maximum capacity: `Q` (from parameters.csv, per truck)
    -   Startup cost: `S` (from parameters.csv, per truck)
    -   Unit transportation cost: `C` (from parameters.csv, per truck)
    -   Customer demand per period: `d1`, `d2`, `d3`, `d4` (from parameters.csv, but identical for all trucks; use once per period)
6.  **Formulate Objective:** Minimize total cost, which is the sum over all trucks and periods of:
    - Startup costs: \( \sum_{i,t} S[i] \cdot u[i,t] \)
    - Transportation costs: \( \sum_{i,t} C[i] \cdot x[i,t] \)
    - Objective: Minimize \( \sum_{i=1}^{10} \sum_{t=1}^{4} (S[i] \cdot u[i,t} + C[i] \cdot x[i,t]) \)
7.  **Formulate Constraints:**
    -   **Demand Satisfaction:** For each period \( t \), total transported weight across all trucks must be at least the customer demand:
        - \( \sum_{i=1}^{10} x[i,t] \geq d_t \) for \( t=1,2,3,4 \)
    -   **Spare-Capacity Buffer:** For each period \( t \), total transported weight cannot exceed 90% of the combined maximum capacity of active trucks:
        - \( \sum_{i=1}^{10} x[i,t] \leq 0.9 \cdot \sum_{i=1}^{10} Q[i] \cdot y[i,t] \)
    -   **Truck Capacity:** For each truck and period, transported weight cannot exceed truck capacity if active, and must be zero if inactive:
        - \( x[i,t] \leq Q[i] \cdot y[i,t] \)
        - \( x[i,t] \geq 0 \)
    -   **Startup Variable Definition:** For each truck and period, startup occurs if truck is on now but was off last period:
        - For \( t=1 \): \( u[i,1] = y[i,1] \) (since all trucks are initially off)
        - For \( t>1 \): \( u[i,t] \geq y[i,t] - y[i,t-1] \), \( u[i,t] \geq 0 \)
    -   **Minimum Up-Time (Once Started, Stay On at Least 2 Periods):**
        - If \( u[i,t]=1 \), then \( y[i,t+1] = 1 \) (for \( t=1,2,3 \); cannot start in period 4)
        - \( u[i,4] = 0 \) (no startups allowed in period 4)
    -   **Minimum Down-Time (After Shut Down, Stay Off at Least 2 Periods):**
        - If truck is on in \( t-1 \) and off in \( t \) (\( y[i,t-1]=1, y[i,t]=0 \)), then \( y[i,t+1]=0 \) (for \( t=1,2 \)), and cannot restart before \( t+2 \)
        - For \( t=3,4 \), adjust for horizon boundary
    -   **Ramping (Change in Load) Constraints:** For each truck and consecutive periods, the change in transported weight cannot exceed 300 kg:
        - For \( t=2,3,4 \): \( |x[i,t] - x[i,t-1]| \leq 300 \)
        - For transitions to/from zero (i.e., truck turning on/off), this applies as well
    -   **No Startup in Period 4:** \( u[i,4] = 0 \)
    -   **Inactive Truck Must Transport Zero:** \( x[i,t] = 0 \) if \( y[i,t]=0 \) (already enforced by \( x[i,t] \leq Q[i] \cdot y[i,t] \))
    -   **Variable Domains:** \( y[i,t], u[i,t] \in \{0,1\} \), \( x[i,t] \geq 0 \)
[Abstract Model Plan END]