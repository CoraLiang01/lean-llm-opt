[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal schedule for activating, starting up, and assigning transported weights to 10 candidate trucks over 4 consecutive periods, so as to meet period-by-period customer demand (with a 10% spare-capacity buffer), while minimizing total startup and transportation costs. The schedule must respect truck-specific capacity, startup cost, and unit transport cost, as well as minimum up/down time, ramping (change in load), and activation rules.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed-charge (startup) and ramping constraints.
3.  **Define Index Sets:** The primary indices are:
    - Trucks: \( i \in \{1, ..., 10\} \) (from 'truck_id')
    - Periods: \( t \in \{1, 2, 3, 4\} \)
4.  **Define Decision Variables:**
    -   `on[i, t]` = 1 if truck \(i\) is active (on) in period \(t\), 0 otherwise. Type: GRB.BINARY.
    -   `startup[i, t]` = 1 if truck \(i\) is started up at the beginning of period \(t\), 0 otherwise. Type: GRB.BINARY.
    -   `shut[i, t]` = 1 if truck \(i\) is shut down at the beginning of period \(t\), 0 otherwise. Type: GRB.BINARY.
    -   `w[i, t]` = weight transported by truck \(i\) in period \(t\). Type: GRB.CONTINUOUS, lower bound 0.
5.  **Identify Parameters (from Schema):**
    -   Truck maximum capacity: `Q` (per truck, column 'Q')
    -   Startup cost: `S` (per truck, column 'S')
    -   Unit transportation cost: `C` (per truck, column 'C')
    -   Period demands: `d1`, `d2`, `d3`, `d4` (columns 'd1'–'d4'; same for all trucks)
6.  **Formulate Objective:** Minimize total cost, which is the sum over all trucks and periods of:
    -   Startup costs: \( \sum_{i, t} S[i] \cdot startup[i, t] \)
    -   Transportation costs: \( \sum_{i, t} C[i] \cdot w[i, t] \)
    -   Objective: Minimize \( \sum_{i, t} S[i] \cdot startup[i, t] + \sum_{i, t} C[i] \cdot w[i, t] \)
7.  **Formulate Constraints:**
    -   **Demand Satisfaction:** For each period \(t\), the total transported weight must be at least the customer demand:
        - \( \sum_{i} w[i, t] \geq d_t \)  (where \(d_t\) is from 'd1', 'd2', etc.)
    -   **Spare-Capacity Buffer:** For each period \(t\), the total transported weight cannot exceed 90% of the combined maximum capacity of active trucks:
        - \( \sum_{i} w[i, t] \leq 0.9 \cdot \sum_{i} Q[i] \cdot on[i, t] \)
    -   **Truck Capacity:** For each truck \(i\) and period \(t\), transported weight cannot exceed truck capacity if active, and must be zero if inactive:
        - \( 0 \leq w[i, t] \leq Q[i] \cdot on[i, t] \)
    -   **Startup Logic:** For each truck \(i\) and period \(t\), startup occurs if the truck is off in \(t-1\) and on in \(t\):
        - \( startup[i, t] \geq on[i, t] - on[i, t-1] \) (with \(on[i, 0] = 0\) since all trucks are initially off)
    -   **Shutdown Logic:** For each truck \(i\) and period \(t\), shutdown occurs if the truck is on in \(t-1\) and off in \(t\):
        - \( shut[i, t] \geq on[i, t-1] - on[i, t] \)
    -   **Minimum Up-Time:** Once a truck is started, it must remain on for at least two consecutive periods. Therefore, if \(startup[i, t] = 1\), then \(on[i, t+1] = 1\) (for \(t = 1, 2, 3\)). No startup allowed in period 4.
        - \( startup[i, 4] = 0 \)
        - For \(t = 1, 2, 3\): \( on[i, t+1] \geq startup[i, t] \)
    -   **Minimum Down-Time:** If a truck is shut down in period \(t\), it must remain off in periods \(t\) and \(t+1\), and cannot be restarted before \(t+2\):
        - For \(t = 1, 2, 3\): \( on[i, t] + on[i, t+1] \leq 2 - shut[i, t] \)
        - For \(t = 4\): no shutdown constraint needed (since horizon ends)
    -   **No Startup in Period 4:** \( startup[i, 4] = 0 \)
    -   **Ramping (Change in Load):** For each truck \(i\) and periods \(t = 2, 3, 4\), the change in transported weight between adjacent periods cannot exceed 300 kg (including transitions to/from zero):
        - \( |w[i, t] - w[i, t-1]| \leq 300 \)
    -   **Initial Conditions:** All trucks are off before period 1:
        - \( on[i, 0] = 0 \) for all \(i\)
        - \( w[i, 0] = 0 \) for all \(i\)
[Abstract Model Plan END]