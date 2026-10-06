[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal activation, startup, and transported-weight schedule for 10 candidate trucks over 4 periods to meet customer demand (with a 10% spare-capacity buffer), minimize total startup and transportation costs, and respect operational constraints such as minimum up/down times, ramping limits, and truck capacities.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed-charge (startup) and ramping constraints.
3.  **Define Index Sets:** The primary indices are:
    - Trucks: \( i \in \{1, ..., 10\} \) (from 'truck_id')
    - Periods: \( t \in \{1, 2, 3, 4\} \)
4.  **Define Decision Variables:**
    -   `x[i, t]` = Amount of weight (kg) transported by truck \(i\) in period \(t\). Type: GRB.CONTINUOUS, \( x[i, t] \geq 0 \).
    -   `y[i, t]` = 1 if truck \(i\) is active (on) in period \(t\), 0 otherwise. Type: GRB.BINARY.
    -   `z[i, t]` = 1 if truck \(i\) is started up (turned on) at the start of period \(t\), 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Truck maximum capacity: `Q` (per truck, from 'Q' column).
    -   Startup cost: `S` (per truck, from 'S' column).
    -   Unit transportation cost: `C` (per truck, from 'C' column).
    -   Customer demand per period: `d1`, `d2`, `d3`, `d4` (from columns 'd1' to 'd4'; same for all trucks).
6.  **Formulate Objective:** Minimize total cost, which is the sum over all trucks and periods of:
    - Startup costs: \( \sum_{i, t} S[i] \cdot z[i, t] \)
    - Transportation costs: \( \sum_{i, t} C[i] \cdot x[i, t] \)
    So, the objective is: Minimize \( \sum_{i, t} S[i] \cdot z[i, t] + C[i] \cdot x[i, t] \).
7.  **Formulate Constraints:**
    -   **Demand Satisfaction:** For each period \(t\), the total transported weight must meet or exceed customer demand:
        - \( \sum_{i} x[i, t] \geq d_t \)  (where \(d_t\) is the demand in period \(t\))
    -   **Spare-Capacity Buffer:** For each period \(t\), the total transported weight cannot exceed 90% of the combined maximum capacity of active trucks:
        - \( \sum_{i} x[i, t] \leq 0.9 \cdot \sum_{i} Q[i] \cdot y[i, t] \)
    -   **Truck Capacity:** For each truck \(i\) and period \(t\), transported weight cannot exceed truck capacity if active, and must be zero if inactive:
        - \( x[i, t] \leq Q[i] \cdot y[i, t] \)
    -   **Activation/Startup Logic:** For each truck \(i\) and period \(t\), define startup variable:
        - \( z[i, t] \geq y[i, t] - y[i, t-1] \) (with \(y[i, 0] = 0\), since all trucks are initially off)
        - \( z[i, t] \in \{0,1\} \), and \( z[i, 4] = 0 \) (cannot start in period 4)
    -   **Minimum Up-Time:** Once a truck is started, it must remain active for at least two consecutive periods:
        - For all \(i\) and \(t \in \{1,2,3\}\): \( y[i, t+1] \geq z[i, t] \)
        - For \(t=4\), cannot start up: \( z[i, 4] = 0 \)
    -   **Minimum Down-Time:** If a truck is shut down (i.e., \(y[i, t-1]=1, y[i, t]=0\)), it must remain off for at least two periods:
        - For all \(i\) and \(t \in \{1,2\}\): \( y[i, t] + y[i, t+1] \leq 1 - (y[i, t-1] - y[i, t]) \)
        - Alternatively, for all \(i\) and \(t \in \{1,2\}\): If \(y[i, t-1]=1, y[i, t]=0\), then \(y[i, t+1]=0\)
        - For \(t=3\), if shut down at \(t=3\), must be off at \(t=4\)
    -   **No Startup in Last Period:** For all \(i\): \( z[i, 4] = 0 \)
    -   **Initial State:** For all \(i\): \( y[i, 0] = 0 \) (trucks are initially off)
    -   **Ramping (Change in Load):** For each truck \(i\) and periods \(t=2,3,4\), the change in transported weight between adjacent periods cannot exceed 300 kg:
        - \( |x[i, t] - x[i, t-1]| \leq 300 \)
        - For \(t=1\), \(x[i, 0]=0\) (since trucks are off before period 1)
    -   **Inactive Truck Must Transport Zero:** For all \(i, t\): \( x[i, t] = 0 \) if \( y[i, t] = 0 \) (already enforced by capacity constraint)
[Abstract Model Plan END]