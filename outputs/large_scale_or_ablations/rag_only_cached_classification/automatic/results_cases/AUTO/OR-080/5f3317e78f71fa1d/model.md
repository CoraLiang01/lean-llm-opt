[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal activation, startup, and transported-weight schedule for 10 candidate trucks over 4 consecutive periods to meet customer demand (with a 10% spare-capacity buffer), while minimizing total startup and transportation costs. The model must respect truck capacity, startup costs, minimum up/down time, ramping (change in load), and buffer constraints.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed-charge (startup) and ramping (change) constraints.
3.  **Define Index Sets:** The primary indices are:
    - Trucks: \( i \in \{1, ..., 10\} \) (from 'truck_id')
    - Periods: \( t \in \{1, 2, 3, 4\} \)
4.  **Define Decision Variables:**
    -   `on[i, t]` = 1 if truck \(i\) is active (on) in period \(t\), 0 otherwise. Type: GRB.BINARY.
    -   `startup[i, t]` = 1 if truck \(i\) is started up at the beginning of period \(t\), 0 otherwise. Type: GRB.BINARY.
    -   `x[i, t]` = amount (kg) transported by truck \(i\) in period \(t\). Type: GRB.CONTINUOUS, \(\geq 0\).
5.  **Identify Parameters (from Schema):**
    -   Maximum capacity per truck: 'Q'
    -   Startup cost per truck: 'S'
    -   Unit transportation cost per truck: 'C'
    -   Customer demand per period: 'd1', 'd2', 'd3', 'd4' (same for all trucks, so use once per period)
6.  **Formulate Objective:** Minimize total cost, which is the sum over all trucks and periods of:
    -   Startup costs: \(\sum_{i, t} S[i] \cdot startup[i, t]\)
    -   Transportation costs: \(\sum_{i, t} C[i] \cdot x[i, t]\)
    -   Objective: Minimize \(\sum_{i, t} S[i] \cdot startup[i, t] + \sum_{i, t} C[i] \cdot x[i, t]\)
7.  **Formulate Constraints:**
    -   **Demand Satisfaction:** For each period \(t\), the total transported weight must be at least the customer demand:
        - \(\sum_{i} x[i, t] \geq d_t\) (where \(d_t\) is 'd1', 'd2', etc.)
    -   **Spare-Capacity Buffer:** For each period \(t\), the total transported weight cannot exceed 90% of the combined maximum capacity of active trucks:
        - \(\sum_{i} x[i, t] \leq 0.9 \cdot \sum_{i} Q[i] \cdot on[i, t]\)
    -   **Truck Capacity:** For each truck \(i\) and period \(t\), transported weight cannot exceed truck capacity if active, and must be zero if inactive:
        - \(x[i, t] \leq Q[i] \cdot on[i, t]\)
        - \(x[i, t] \geq 0\)
    -   **Startup Definition:** For each truck \(i\) and period \(t\), startup occurs if truck is on in \(t\) but was off in \(t-1\):
        - \(startup[i, t] \geq on[i, t] - on[i, t-1]\) (with \(on[i, 0] = 0\) since all trucks are initially off)
    -   **Minimum Up-Time:** Once a truck is started, it must remain on for at least two consecutive periods. Therefore, if \(startup[i, t] = 1\), then \(on[i, t+1] = 1\) (for \(t = 1, 2, 3\)), and no startup is allowed in period 4.
    -   **No Startup in Period 4:** For all trucks, \(startup[i, 4] = 0\).
    -   **Minimum Down-Time:** If a truck is shut down (i.e., \(on[i, t-1]=1\), \(on[i, t]=0\)), then it must remain off in periods \(t\) and \(t+1\), and cannot be restarted before period \(t+2\). For all applicable \(t\), enforce:
        - If \(on[i, t-1]=1\) and \(on[i, t]=0\), then \(on[i, t+1]=0\).
        - (No restart before \(t+2\) is automatically enforced by startup definition and minimum up-time.)
    -   **Ramping (Change in Load):** For each truck and consecutive periods, the change in transported weight (including to/from zero) cannot exceed 300 kg:
        - \(|x[i, t] - x[i, t-1]| \leq 300\) for \(t=2,3,4\), with \(x[i, 0]=0\).
    -   **Inactive Truck Carries No Load:** For all \(i, t\), if \(on[i, t]=0\), then \(x[i, t]=0\) (already enforced by capacity constraint).
[Abstract Model Plan END]