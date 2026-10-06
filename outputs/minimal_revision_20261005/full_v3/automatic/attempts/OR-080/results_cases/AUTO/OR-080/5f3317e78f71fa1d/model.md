[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal activation, startup, and transported-weight schedule for 10 candidate trucks over 4 periods to meet customer demand (with a 10% spare-capacity buffer), minimize total startup and transportation costs, and respect operational constraints such as minimum up/down times, ramping limits, and truck capacities.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed-charge (startup) and ramping constraints.
3.  **Define Index Sets:** The primary indices are:
    - Trucks: \( i \in \{1, 2, ..., 10\} \) (from 'truck_id')
    - Periods: \( t \in \{1, 2, 3, 4\} \)
4.  **Define Decision Variables:**
    -   `x[i, t]` = Amount of goods (kg) transported by truck \(i\) in period \(t\). Type: GRB.CONTINUOUS, \( x[i, t] \geq 0 \).
    -   `y[i, t]` = 1 if truck \(i\) is active (on) in period \(t\), 0 otherwise. Type: GRB.BINARY.
    -   `z[i, t]` = 1 if truck \(i\) is started up (turned on) at the start of period \(t\), 0 otherwise. Type: GRB.BINARY.
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
    -   **Demand Satisfaction:** For each period \(t\), the total transported weight must meet or exceed customer demand:
        - \( \sum_{i=1}^{10} x[i, t] \geq d_t \) (where \(d_t\) is the demand in period \(t\))
    -   **Spare-Capacity Buffer:** For each period \(t\), total transported weight cannot exceed 90% of the combined capacity of active trucks:
        - \( \sum_{i=1}^{10} x[i, t] \leq 0.9 \times \sum_{i=1}^{10} Q[i] \cdot y[i, t] \)
    -   **Truck Capacity:** For each truck and period, transported weight cannot exceed truck capacity if active, and must be zero if inactive:
        - \( x[i, t] \leq Q[i] \cdot y[i, t] \)
        - \( x[i, t] \geq 0 \)
    -   **Startup Variable Definition:** For each truck and period, startup occurs if truck is on in \(t\) but was off in \(t-1\):
        - For \(t=1\): \( z[i, 1] \geq y[i, 1] \) (since all trucks are initially off)
        - For \(t>1\): \( z[i, t] \geq y[i, t] - y[i, t-1] \)
    -   **Minimum Up-Time (at least 2 consecutive periods):** If a truck is started in period \(t\), it must remain on in period \(t+1\):
        - For \(t=1,2,3\): \( y[i, t+1] \geq z[i, t] \)
        - No startups allowed in period 4: \( z[i, 4] = 0 \)
    -   **Minimum Down-Time (if shut down, must stay off for 2 periods):** If a truck is turned off in period \(t\) after being on in \(t-1\), it must be off in \(t\) and \(t+1\):
        - For \(t=1,2,3\): \( y[i, t-1] - y[i, t] \leq 1 - y[i, t+1] \) (or equivalent logic to enforce two-period downtime)
        - If a truck is off in \(t\) after being on in \(t-1\), then \( y[i, t+1] = 0 \)
    -   **No Startup in Period 4:** \( z[i, 4] = 0 \)
    -   **Initial State:** All trucks are off before period 1 (\( y[i, 0] = 0 \) for logic).
    -   **Ramping Constraint (change in transported weight):** For each truck and adjacent periods, the absolute change in transported weight cannot exceed 300 kg:
        - For \(t=2,3,4\): \( |x[i, t] - x[i, t-1]| \leq 300 \)
    -   **Inactive Truck Must Transport Zero:** For all \(i, t\): \( x[i, t] = 0 \) if \( y[i, t] = 0 \) (already enforced by \( x[i, t] \leq Q[i] \cdot y[i, t] \))
[Abstract Model Plan END]