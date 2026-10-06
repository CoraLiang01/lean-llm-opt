[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal activation, startup, and transported-weight schedule for 10 candidate trucks over 4 consecutive periods to meet customer demand (with a 10% spare-capacity buffer), while minimizing total startup and transportation costs. The model must respect truck startup/shutdown rules, minimum up/down times, ramping (change in load) limits, and per-truck capacity constraints.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed-charge (startup) and variable (transportation) costs, minimum up/down time, and ramping constraints.
3.  **Define Index Sets:** The primary indices are:
    - Trucks: \( i \in \{1, ..., 10\} \) (from 'truck_id' in parameters.csv)
    - Periods: \( t \in \{1, 2, 3, 4\} \)
4.  **Define Decision Variables:**
    -   `x[i, t]` = Amount of goods (kg) transported by truck \(i\) in period \(t\). Type: GRB.CONTINUOUS, \( x[i, t] \geq 0 \).
    -   `y[i, t]` = 1 if truck \(i\) is active (on) in period \(t\), 0 otherwise. Type: GRB.BINARY.
    -   `z[i, t]` = 1 if truck \(i\) is started up at the beginning of period \(t\), 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Truck maximum capacity: `Q` (kg) for each truck \(i\).
    -   Truck startup cost: `S` for each truck \(i\).
    -   Truck unit transportation cost: `C` for each truck \(i\).
    -   Customer demand per period: `d1`, `d2`, `d3`, `d4` (same for all trucks, used as demand in each period).
6.  **Formulate Objective:** Minimize total cost, which is the sum over all trucks and periods of:
    - Startup costs: \( \sum_{i, t} S[i] \cdot z[i, t] \)
    - Transportation costs: \( \sum_{i, t} C[i] \cdot x[i, t] \)
    - Objective: Minimize \( \sum_{i, t} S[i] \cdot z[i, t] + \sum_{i, t} C[i] \cdot x[i, t] \)
7.  **Formulate Constraints:**
    -   **Demand Satisfaction:** For each period \(t\), the total transported weight must meet or exceed customer demand:
        - \( \sum_{i} x[i, t] \geq d_t \) (where \(d_t\) is `d1`, `d2`, `d3`, or `d4` for \(t=1,2,3,4\))
    -   **Spare-Capacity Buffer (90% Rule):** For each period \(t\), the total transported weight cannot exceed 90% of the combined maximum capacity of active trucks:
        - \( \sum_{i} x[i, t] \leq 0.9 \cdot \sum_{i} Q[i] \cdot y[i, t] \)
    -   **Truck Capacity:** For each truck \(i\) and period \(t\), transported weight cannot exceed truck capacity if active, and must be zero if inactive:
        - \( x[i, t] \leq Q[i] \cdot y[i, t] \)
        - \( x[i, t] \geq 0 \)
    -   **Startup Logic:** For each truck \(i\) and period \(t\), startup variable is 1 if truck is turned on in \(t\) after being off in \(t-1\):
        - For \(t=1\): \( z[i, 1] \geq y[i, 1] \) (since all trucks are initially off)
        - For \(t>1\): \( z[i, t] \geq y[i, t] - y[i, t-1] \)
    -   **Minimum Up-Time (at least 2 consecutive periods):** If a truck is started in period \(t\), it must remain active in period \(t\) and \(t+1\):
        - For \(t=1,2,3\): \( y[i, t+1] \geq z[i, t] \)
        - No startup allowed in period 4: \( z[i, 4] = 0 \)
    -   **Minimum Down-Time (at least 2 consecutive periods off after shutdown):** If a truck is shut down in period \(t\) (i.e., \(y[i, t-1]=1, y[i, t]=0\)), it must remain off in periods \(t\) and \(t+1\), and cannot be restarted before \(t+2\):
        - For \(t=2,3\): \( y[i, t] + y[i, t+1] \leq 1 \) if \(y[i, t-1]=1\) and \(y[i, t]=0\)
        - Alternatively, enforce: \( y[i, t-1] - y[i, t] \leq 1 - y[i, t+1] \) and \( y[i, t+1] \leq 1 - (y[i, t-1] - y[i, t]) \)
        - For \(t=4\): No shutdown constraint needed as horizon ends.
    -   **No Startup in Last Period:** \( z[i, 4] = 0 \) for all \(i\).
    -   **Initial State:** All trucks are off before period 1: \( y[i, 0] = 0 \) (implicitly for startup logic).
    -   **Ramping (Change in Load) Constraint:** For each truck \(i\) and periods \(t=2,3,4\), the change in transported weight between adjacent periods cannot exceed 300 kg:
        - \( |x[i, t] - x[i, t-1]| \leq 300 \)
        - For \(t=1\), previous period is 0 (since all trucks are off): \( |x[i, 1] - 0| \leq 300 \)
    -   **Inactive Truck Must Transport Zero:** For all \(i, t\): \( x[i, t] = 0 \) if \( y[i, t] = 0 \) (already enforced by \( x[i, t] \leq Q[i] \cdot y[i, t] \))
[Abstract Model Plan END]