[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal activation, startup, and transported-weight schedule for 10 candidate trucks over 4 consecutive periods to meet period-specific customer demands, while minimizing total startup and transportation costs. The solution must respect truck capacity, minimum up/down time, ramping (change in load), and spare-capacity buffer constraints.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed-charge (startup) and ramping constraints.
3.  **Define Index Sets:** The primary indices are:
    - Trucks: \( i \in \{\text{truck\_id}\} \) (from 1 to 10)
    - Periods: \( t \in \{1, 2, 3, 4\} \)
4.  **Define Decision Variables:**
    - \( y_{i,t} \) = 1 if truck \( i \) is active (on) in period \( t \), 0 otherwise. Type: GRB.BINARY.
    - \( s_{i,t} \) = 1 if truck \( i \) is started up at the beginning of period \( t \), 0 otherwise. Type: GRB.BINARY.
    - \( z_{i,t} \) = 1 if truck \( i \) is shut down at the beginning of period \( t \), 0 otherwise. Type: GRB.BINARY.
    - \( x_{i,t} \) = weight transported by truck \( i \) in period \( t \) (kg). Type: GRB.CONTINUOUS.
5.  **Identify Parameters (from Schema):**
    - Truck maximum capacity: \( Q_i \) (from column 'Q')
    - Startup cost: \( S_i \) (from column 'S')
    - Unit transportation cost: \( C_i \) (from column 'C')
    - Period demands: \( d_t \) (from columns 'd1', 'd2', 'd3', 'd4')
6.  **Formulate Objective:** Minimize total cost, which is the sum over all trucks and periods of:
    - Startup costs: \( S_i \cdot s_{i,t} \)
    - Transportation costs: \( C_i \cdot x_{i,t} \)
    So, Objective = \( \sum_{i} \sum_{t} (S_i \cdot s_{i,t} + C_i \cdot x_{i,t}) \)
7.  **Formulate Constraints:**
    - **Demand Satisfaction:** For each period \( t \), \( \sum_{i} x_{i,t} \geq d_t \)
    - **Spare-Capacity Buffer:** For each period \( t \), \( \sum_{i} x_{i,t} \leq 0.9 \cdot \sum_{i} Q_i \cdot y_{i,t} \)
    - **Truck Capacity:** For all \( i, t \), \( 0 \leq x_{i,t} \leq Q_i \cdot y_{i,t} \)
    - **Activation/Startup Logic:** For all \( i, t \), \( s_{i,t} \geq y_{i,t} - y_{i,t-1} \) (with \( y_{i,0} = 0 \)), and \( s_{i,1} = y_{i,1} \) (since all trucks are initially off)
    - **Minimum Up-Time:** If a truck is started in period \( t \), it must remain on in period \( t \) and \( t+1 \). For all \( i, t \leq 3 \): \( y_{i,t+1} \geq s_{i,t} \). No truck may be started in period 4: \( s_{i,4} = 0 \)
    - **Minimum Down-Time:** If a truck is shut down in period \( t \), it must remain off in periods \( t \) and \( t+1 \), and cannot be restarted before \( t+2 \). For all \( i, t \leq 3 \): \( y_{i,t+1} \leq 1 - z_{i,t} \), \( y_{i,t+2} \leq 1 - z_{i,t} \). Define \( z_{i,t} \geq y_{i,t-1} - y_{i,t} \) (with \( y_{i,0} = 0 \))
    - **No Startup in Last Period:** For all \( i \), \( s_{i,4} = 0 \)
    - **Ramp (Change in Load) Constraint:** For all \( i, t = 2,3,4 \): \( |x_{i,t} - x_{i,t-1}| \leq 300 \)
    - **Inactive Truck Zero Load:** For all \( i, t \), \( x_{i,t} \leq Q_i \cdot y_{i,t} \) and \( x_{i,t} \geq 0 \)
[Abstract Model Plan END]