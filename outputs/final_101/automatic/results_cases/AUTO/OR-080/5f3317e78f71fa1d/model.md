[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal activation, startup, and transported-weight schedule for 10 candidate trucks over 4 periods to meet customer demand (with a 10% spare-capacity buffer), minimize total startup and transportation costs, and respect truck startup/shutdown, minimum up/down time, ramping, and capacity constraints.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed-charge (startup) and ramping constraints.
3.  **Define Index Sets:** The primary indices are:
    - Trucks: \( i \in \{1, ..., 10\} \) (from 'truck_id')
    - Periods: \( t \in \{1, 2, 3, 4\} \)
4.  **Define Decision Variables:**
    -   `x[i, t]` = Amount of goods (kg) transported by truck \(i\) in period \(t\). Type: GRB.CONTINUOUS (≥ 0).
    -   `y[i, t]` = 1 if truck \(i\) is active (on) in period \(t\), 0 otherwise. Type: GRB.BINARY.
    -   `z[i, t]` = 1 if truck \(i\) is started up (turned on) at the start of period \(t\), 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Truck maximum capacity: `Q` (per truck, from 'Q' column).
    -   Startup cost: `S` (per truck, from 'S' column).
    -   Unit transportation cost: `C` (per truck, from 'C' column).
    -   Customer demand per period: `d1`, `d2`, `d3`, `d4` (from columns 'd1'–'d4'; same for all trucks).
6.  **Formulate Objective:** Minimize total cost, which is the sum over all trucks and periods of:
    -   Startup costs: \(\sum_{i,t} S[i] \cdot z[i, t]\)
    -   Transportation costs: \(\sum_{i,t} C[i] \cdot x[i, t]\)
    -   So, Objective: Minimize \(\sum_{i=1}^{10} \sum_{t=1}^{4} [S[i] \cdot z[i, t} + C[i] \cdot x[i, t]]\)
7.  **Formulate Constraints:**
    -   **Demand Satisfaction:** For each period \(t\), total transported weight across all trucks must be at least the customer demand:
        - \(\sum_{i=1}^{10} x[i, t] \geq d_t\) for \(t = 1,2,3,4\)
    -   **Spare-Capacity Buffer:** For each period \(t\), total transported weight cannot exceed 90% of the combined maximum capacity of active trucks:
        - \(\sum_{i=1}^{10} x[i, t] \leq 0.9 \cdot \sum_{i=1}^{10} Q[i] \cdot y[i, t]\)
    -   **Truck Capacity:** For each truck and period, transported weight cannot exceed truck capacity if active, and must be zero if inactive:
        - \(x[i, t] \leq Q[i] \cdot y[i, t]\) for all \(i, t\)
        - \(x[i, t] \geq 0\) for all \(i, t\)
    -   **Startup Logic:** For each truck and period, startup variable is 1 if truck is turned on in period \(t\) after being off in \(t-1\):
        - For \(t=1\): \(z[i,1] = y[i,1]\) (since all trucks are initially off)
        - For \(t>1\): \(z[i,t] \geq y[i,t] - y[i,t-1]\)
    -   **Minimum Up-Time (at least 2 consecutive periods on after startup):**
        - If \(z[i,t]=1\), then \(y[i,t+1]=1\) (for \(t=1,2,3\)), i.e., after startup, must stay on at least one more period.
        - No startup allowed in period 4: \(z[i,4]=0\)
    -   **Minimum Down-Time (if truck is shut down, must stay off for at least 2 periods):**
        - If \(y[i,t-1]=1\) and \(y[i,t]=0\), then \(y[i,t+1]=0\) (for \(t=2,3\)), i.e., after shutdown, must stay off at least one more period.
        - Cannot restart before \(t+2\) after shutdown.
    -   **Ramping Constraint (change in transported weight between adjacent periods ≤ 300 kg):**
        - For all \(i\) and \(t=2,3,4\): \(|x[i, t] - x[i, t-1]| \leq 300\)
        - For \(t=1\), initial value is 0 (since all trucks are off before period 1): \(|x[i,1] - 0| \leq 300\)
    -   **No startup in period 4:** \(z[i,4]=0\) for all \(i\)
    -   **Inactive trucks must transport zero:** \(x[i, t] = 0\) if \(y[i, t]=0\) (already enforced by \(x[i, t] \leq Q[i] \cdot y[i, t]\))
    -   **Variable domains:** \(y[i, t], z[i, t] \in \{0,1\}\); \(x[i, t] \geq 0\)
[Abstract Model Plan END]