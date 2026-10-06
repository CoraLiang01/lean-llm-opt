[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal activation, startup, and transported-weight schedule for 10 candidate trucks over 4 consecutive periods to meet customer demand (with a 10% spare-capacity buffer), while minimizing total startup and transportation costs. The model must respect truck startup/shutdown rules, minimum up/down times, ramping (change in load) limits, and per-truck capacity constraints.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed-charge (startup) and variable (transportation) costs, subject to scheduling and ramping constraints.
3.  **Define Index Sets:** The primary indices are:
    - Trucks: \( i \in \{1,2,\ldots,10\} \) (from 'truck_id' in parameters.csv)
    - Periods: \( t \in \{1,2,3,4\} \) (corresponding to d1, d2, d3, d4)
4.  **Define Decision Variables:**
    -   `x[i,t]` = Amount of goods (kg) transported by truck \(i\) in period \(t\). Type: GRB.CONTINUOUS, \( x[i,t] \geq 0 \).
    -   `y[i,t]` = 1 if truck \(i\) is active (on) in period \(t\), 0 otherwise. Type: GRB.BINARY.
    -   `u[i,t]` = 1 if truck \(i\) is started up at the beginning of period \(t\), 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Truck maximum capacity: `Q` (kg) for each truck \(i\).
    -   Startup cost: `S` for each truck \(i\).
    -   Unit transportation cost: `C` for each truck \(i\).
    -   Customer demand per period: `d1`, `d2`, `d3`, `d4` (same for all trucks, but indexed by period).
6.  **Formulate Objective:** Minimize total cost, which is the sum of all truck startup costs (incurred when a truck is started in any period) plus the sum of all transportation costs (amount transported times unit cost):
    - Minimize: \( \sum_{i=1}^{10} \sum_{t=1}^{4} S_i \cdot u[i,t] + \sum_{i=1}^{10} \sum_{t=1}^{4} C_i \cdot x[i,t] \)
7.  **Formulate Constraints:**
    -   **Demand Satisfaction:** For each period \(t\), the total transported weight must be at least the customer demand:
        - \( \sum_{i=1}^{10} x[i,t] \geq d_t \) for \( t = 1,2,3,4 \)
    -   **Spare-Capacity Buffer (90% Rule):** For each period \(t\), the total transported weight cannot exceed 90% of the combined maximum capacity of active trucks:
        - \( \sum_{i=1}^{10} x[i,t] \leq 0.9 \cdot \sum_{i=1}^{10} Q_i \cdot y[i,t] \) for all \(t\)
    -   **Truck Capacity:** For each truck and period, transported weight cannot exceed truck capacity if active, and must be zero if inactive:
        - \( x[i,t] \leq Q_i \cdot y[i,t] \) for all \(i, t\)
        - \( x[i,t] \geq 0 \) for all \(i, t\)
    -   **Startup Logic:** A truck is started in period \(t\) if it is off in \(t-1\) and on in \(t\):
        - \( u[i,t] \geq y[i,t] - y[i,t-1] \) for \( t=1,2,3,4 \), with \( y[i,0]=0 \) (all trucks off before period 1)
        - \( u[i,t] \in \{0,1\} \)
    -   **Minimum Up-Time (at least 2 consecutive periods):** If a truck is started in period \(t\), it must remain on in period \(t\) and \(t+1\):
        - For \( t=1,2,3 \): \( y[i,t+1] \geq y[i,t] - y[i,t-1] \) (if started at \(t\), must be on at \(t+1\))
        - No truck may be started in period 4: \( u[i,4] = 0 \)
    -   **Minimum Down-Time (at least 2 consecutive periods):** If a truck is shut down in period \(t\) (i.e., on in \(t-1\), off in \(t\)), it must remain off in \(t\) and \(t+1\), and cannot be restarted before \(t+2\):
        - For \( t=1,2,3 \): If \( y[i,t-1]=1, y[i,t]=0 \), then \( y[i,t+1]=0 \)
        - Enforced via: \( y[i,t-1] - y[i,t] \leq 1 - y[i,t+1] \) for \( t=1,2,3 \)
    -   **No Start in Last Period:** \( u[i,4] = 0 \) for all \(i\)
    -   **Ramping (Change in Load) Constraint:** For each truck, the absolute change in transported weight between adjacent periods cannot exceed 300 kg, including transitions to/from zero:
        - \( |x[i,t] - x[i,t-1]| \leq 300 \) for \( t=2,3,4 \), with \( x[i,0]=0 \)
    -   **Inactive Truck Must Transport Zero:** \( x[i,t] = 0 \) whenever \( y[i,t]=0 \)
    -   **Initial State:** All trucks are off before period 1: \( y[i,0]=0 \), \( x[i,0]=0 \)
[Abstract Model Plan END]