[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal activation, startup, and transported-weight schedule for 10 candidate trucks over 4 consecutive periods to meet customer demand (with a 10% spare-capacity buffer), while minimizing total startup and transportation costs. The schedule must respect truck-specific capacity, startup costs, unit transport costs, minimum up/down time, ramping (change in load), and activation rules.
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
    -   Truck maximum capacity: `Q` (per truck)
    -   Startup cost: `S` (per truck)
    -   Unit transportation cost: `C` (per truck)
    -   Customer demand per period: `d1`, `d2`, `d3`, `d4` (identical for all trucks, but used per period)
6.  **Formulate Objective:** Minimize total cost, which is the sum over all trucks and periods of:
    -   Startup costs: \( \sum_{i,t} S[i] \cdot startup[i, t] \)
    -   Transportation costs: \( \sum_{i,t} C[i] \cdot w[i, t] \)
    -   Objective: Minimize \( \sum_{i,t} S[i] \cdot startup[i, t] + \sum_{i,t} C[i] \cdot w[i, t] \)
7.  **Formulate Constraints:**
    -   **Demand Satisfaction:** For each period \(t\), the total transported weight must be at least the customer demand:
        - \( \sum_{i} w[i, t] \geq d_t \) (where \(d_t\) is from `d1`, `d2`, `d3`, `d4`)
    -   **Spare-Capacity Buffer:** For each period \(t\), the total transported weight cannot exceed 90% of the combined maximum capacity of active trucks:
        - \( \sum_{i} w[i, t] \leq 0.9 \cdot \sum_{i} Q[i] \cdot on[i, t] \)
    -   **Truck Capacity:** For each truck and period, transported weight cannot exceed truck capacity if active, and must be zero if inactive:
        - \( w[i, t] \leq Q[i] \cdot on[i, t] \)
        - \( w[i, t] \geq 0 \)
    -   **Startup Definition:** Startup occurs if a truck is off in \(t-1\) and on in \(t\):
        - \( startup[i, t] \geq on[i, t] - on[i, t-1] \) (with \(on[i, 0] = 0\) since all trucks are initially off)
    -   **No Startup in Period 4:** No truck can be started in period 4:
        - \( startup[i, 4] = 0 \)
    -   **Minimum Up-Time (2 periods):** Once a truck is started, it must remain on for at least 2 consecutive periods:
        - If \( startup[i, t] = 1 \), then \( on[i, t+1] = 1 \) (for \(t = 1, 2, 3\))
    -   **Minimum Down-Time (2 periods):** If a truck is shut down after being on, it must stay off for at least 2 periods:
        - Define \( shut[i, t] \geq on[i, t-1] - on[i, t] \) (for \(t \geq 1\))
        - If \( shut[i, t] = 1 \), then \( on[i, t+1] = 0 \) and \( on[i, t+2] = 0 \) (for applicable \(t\))
        - A truck cannot be restarted before \(t+2\) after shutdown.
    -   **No Startup in Period 4:** (repeated for clarity) \( startup[i, 4] = 0 \)
    -   **Ramping Constraint:** For each truck and consecutive periods, the change in transported weight (including to/from zero) cannot exceed 300 kg:
        - \( |w[i, t] - w[i, t-1]| \leq 300 \) (with \(w[i, 0] = 0\))
    -   **Inactive Truck Carries No Load:** If \( on[i, t] = 0 \), then \( w[i, t] = 0 \)
        - Already enforced by \( w[i, t] \leq Q[i] \cdot on[i, t] \)
    -   **Initial State:** All trucks are off before period 1 (\( on[i, 0] = 0 \)), and \( w[i, 0] = 0 \).
[Abstract Model Plan END]