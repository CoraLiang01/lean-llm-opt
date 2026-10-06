[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal activation, startup, and transported-weight schedule for 10 candidate trucks over 4 periods to meet customer demand (with a 10% spare-capacity buffer), while minimizing total startup and transportation costs. The model must respect truck startup/shutdown rules, minimum up/down times, per-truck ramping (change in load), and per-period demand and capacity constraints.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed-charge (startup) and ramping constraints.
3.  **Define Index Sets:** The primary indices are:
    - Trucks: \( i \in \{1, ..., 10\} \) (from 'truck_id')
    - Periods: \( t \in \{1, 2, 3, 4\} \)
4.  **Define Decision Variables:**
    -   `x[i, t]` = Amount of goods (kg) transported by truck \(i\) in period \(t\). Type: GRB.CONTINUOUS (≥ 0).
    -   `y[i, t]` = 1 if truck \(i\) is active (on) in period \(t\), 0 otherwise. Type: GRB.BINARY.
    -   `z[i, t]` = 1 if truck \(i\) is started up at the beginning of period \(t\), 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Truck maximum capacity: `Q` (per truck, from 'Q' column).
    -   Startup cost: `S` (per truck, from 'S' column).
    -   Unit transportation cost: `C` (per truck, from 'C' column).
    -   Customer demand per period: `d1`, `d2`, `d3`, `d4` (from columns 'd1'–'d4'; same for all trucks).
6.  **Formulate Objective:** Minimize total cost, which is the sum over all trucks and periods of:
    - Startup costs: \( \sum_{i,t} S_i \cdot z[i, t] \)
    - Transportation costs: \( \sum_{i,t} C_i \cdot x[i, t] \)
    - Objective: Minimize \( \sum_{i=1}^{10} \sum_{t=1}^{4} (S_i \cdot z[i, t] + C_i \cdot x[i, t]) \)
7.  **Formulate Constraints:**
    -   **Demand Satisfaction:** For each period \(t\), the total transported weight must meet or exceed customer demand:
        - \( \sum_{i=1}^{10} x[i, t] \geq d_t \)  (where \(d_t\) is from 'd1', 'd2', etc.)
    -   **Spare-Capacity Buffer:** For each period \(t\), total transported weight cannot exceed 90% of the combined capacity of active trucks:
        - \( \sum_{i=1}^{10} x[i, t] \leq 0.9 \cdot \sum_{i=1}^{10} Q_i \cdot y[i, t] \)
    -   **Truck Capacity:** For each truck and period, transported weight cannot exceed truck capacity if active, and must be zero if inactive:
        - \( x[i, t] \leq Q_i \cdot y[i, t] \)
        - \( x[i, t] \geq 0 \)
    -   **Startup Definition:** For each truck and period, startup occurs if truck is off in \(t-1\) and on in \(t\):
        - \( z[i, t] \geq y[i, t] - y[i, t-1] \) (with \(y[i, 0] = 0\) since all trucks are initially off)
        - \( z[i, t] \leq y[i, t] \)
        - \( z[i, t] \leq 1 - y[i, t-1] \)
        - No startups allowed in period 4: \( z[i, 4] = 0 \)
    -   **Minimum Up-Time:** Once a truck is started, it must remain active for at least two consecutive periods:
        - For all \(i\) and \(t \in \{1,2,3\}\): If \(z[i, t] = 1\), then \(y[i, t+1] = 1\)
        - No startups in period 4 (already above)
    -   **Minimum Down-Time:** If a truck is shut down (on in \(t-1\), off in \(t\)), it must remain off in \(t\) and \(t+1\), and cannot be restarted before \(t+2\):
        - For all \(i\) and \(t \in \{1,2\}\): If \(y[i, t-1] = 1\) and \(y[i, t] = 0\), then \(y[i, t+1] = 0\)
        - For all \(i\) and \(t \in \{1,2\}\): \( y[i, t-1] - y[i, t] \leq 1 - y[i, t+1] \)
        - For all \(i\) and \(t \in \{1,2\}\): \( z[i, t+1] \leq y[i, t] \) (no restart in \(t+1\) after shutdown)
    -   **Ramping Constraint:** For each truck and adjacent periods, the change in transported weight (including to/from zero) cannot exceed 300 kg:
        - For all \(i\) and \(t \in \{2,3,4\}\): \( |x[i, t] - x[i, t-1]| \leq 300 \)
    -   **Initial State:** All trucks are off before period 1: \( y[i, 0] = 0 \), \( x[i, 0] = 0 \)
    -   **Non-negativity and Binary:** All \(x[i, t] \geq 0\); all \(y[i, t], z[i, t] \in \{0,1\}\)
[Abstract Model Plan END]