[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal activation, startup, and transported-weight schedule for 10 candidate trucks over 4 consecutive periods to meet customer demand (with a 10% spare-capacity buffer), while minimizing total startup and transportation costs. The model must respect truck startup/shutdown rules, minimum up/down times, ramping (change in load) limits, and per-truck capacity constraints.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed-charge (startup) and variable (transportation) costs, minimum up/down time, and ramping constraints.
3.  **Define Index Sets:** The primary indices are:
    - Trucks: \( i \in \{1, ..., 10\} \) (from 'truck_id' in parameters.csv)
    - Periods: \( t \in \{1, 2, 3, 4\} \)
4.  **Define Decision Variables:**
    -   `x[i, t]` = Amount of goods (kg) transported by truck \(i\) in period \(t\). Type: GRB.CONTINUOUS, \( x[i, t] \geq 0 \).
    -   `y[i, t]` = 1 if truck \(i\) is active (on) in period \(t\), 0 otherwise. Type: GRB.BINARY.
    -   `u[i, t]` = 1 if truck \(i\) is started up at the beginning of period \(t\), 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Truck maximum capacity: `Q` (kg) for each truck \(i\).
    -   Startup cost: `S` for each truck \(i\).
    -   Unit transportation cost: `C` for each truck \(i\).
    -   Customer demand per period: `d1`, `d2`, `d3`, `d4` (same for all trucks, but indexed by period).
6.  **Formulate Objective:** Minimize total cost, which is the sum of all truck startup costs (incurred when a truck is started in any period) plus the sum of all transportation costs (amount transported times unit cost for each truck and period):
    - Minimize: \( \sum_{i} \sum_{t} S[i] \cdot u[i, t] + \sum_{i} \sum_{t} C[i] \cdot x[i, t] \)
7.  **Formulate Constraints:**
    -   **Demand Satisfaction:** For each period \(t\), the total transported weight across all trucks must be at least the customer demand for that period.
        - \( \sum_{i} x[i, t] \geq d_t \) for all \( t \).
    -   **Spare-Capacity Buffer (Upper Bound):** In each period, the total transported weight cannot exceed 90% of the combined maximum capacity of all active trucks.
        - \( \sum_{i} x[i, t] \leq 0.9 \cdot \sum_{i} Q[i] \cdot y[i, t] \) for all \( t \).
    -   **Truck Capacity:** For each truck and period, transported weight cannot exceed the truck's capacity if active, and must be zero if inactive.
        - \( x[i, t] \leq Q[i] \cdot y[i, t] \) for all \( i, t \).
        - \( x[i, t] \geq 0 \) for all \( i, t \).
    -   **Startup Logic:** A truck is started up in period \(t\) if it is active in \(t\) but was inactive in \(t-1\) (with all trucks off before period 1).
        - \( u[i, t] \geq y[i, t] - y[i, t-1] \) for all \( i, t \) (define \( y[i, 0] = 0 \)).
    -   **Minimum Up-Time:** Once a truck is started, it must remain active for at least two consecutive periods. No truck may be started in period 4 (since two periods are required).
        - For all \( i \) and \( t \in \{1, 2, 3\} \): If \( u[i, t] = 1 \), then \( y[i, t+1] = 1 \).
        - \( u[i, 4] = 0 \) for all \( i \).
    -   **Minimum Down-Time:** If a truck is shut down (i.e., \( y[i, t-1]=1, y[i, t]=0 \)), it must remain off in periods \(t\) and \(t+1\), and cannot be restarted before \(t+2\).
        - For all \( i \) and \( t \in \{1, 2, 3\} \): If \( y[i, t-1]=1, y[i, t]=0 \), then \( y[i, t+1]=0 \).
        - (No wrap-around: period 4 is the last period.)
    -   **Initial State:** All trucks are off before period 1.
        - \( y[i, 0] = 0 \) for all \( i \).
    -   **Ramping (Change in Load):** For each truck, the absolute change in transported weight between consecutive periods cannot exceed 300 kg, including transitions to or from zero.
        - \( |x[i, t] - x[i, t-1]| \leq 300 \) for all \( i, t \in \{2, 3, 4\} \), with \( x[i, 0] = 0 \).
    -   **Inactive Truck Must Transport Zero:** If \( y[i, t]=0 \), then \( x[i, t]=0 \) (already enforced by capacity constraint).
[Abstract Model Plan END]