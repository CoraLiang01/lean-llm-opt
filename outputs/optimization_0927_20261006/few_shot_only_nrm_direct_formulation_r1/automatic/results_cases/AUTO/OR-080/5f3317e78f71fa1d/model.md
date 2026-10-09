[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal activation, startup, and transported-weight schedule for 10 candidate trucks over 4 consecutive periods to meet customer demand (with a 10% spare-capacity buffer), minimize total startup and transportation costs, and respect truck-specific capacity, startup, ramping, and minimum up/down time constraints.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed-charge (startup) and ramping/minimum up/down time constraints.
3.  **Define Index Sets:** The primary indices are:
    - Trucks: \( i \in \{\text{truck 1}, \ldots, \text{truck 10}\} \) (from all rows of parameters.csv, using 'truck_id')
    - Periods: \( t \in \{1,2,3,4\} \)
4.  **Define Decision Variables:**
    - \( x_{i,t} \) = Amount of weight (kg) transported by truck \( i \) in period \( t \). Type: GRB.CONTINUOUS, \( x_{i,t} \geq 0 \).
    - \( y_{i,t} \) = 1 if truck \( i \) is active (on) in period \( t \), 0 otherwise. Type: GRB.BINARY.
    - \( s_{i,t} \) = 1 if truck \( i \) is started up at the beginning of period \( t \), 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    - Truck maximum capacity: \( Q_i \) (from 'Q' in parameters.csv)
    - Startup cost: \( S_i \) (from 'S')
    - Unit transportation cost: \( C_i \) (from 'C')
    - Customer demand per period: \( d_t \) (from 'd1', 'd2', 'd3', 'd4'; same for all trucks)
6.  **Formulate Objective:** Minimize total cost, which is the sum over all trucks and periods of:
    - Startup costs: \( \sum_{i,t} S_i \cdot s_{i,t} \)
    - Transportation costs: \( \sum_{i,t} C_i \cdot x_{i,t} \)
    So, objective: Minimize \( \sum_{i} \sum_{t} (S_i \cdot s_{i,t} + C_i \cdot x_{i,t}) \)
7.  **Formulate Constraints:**
    - **Demand Satisfaction:** For each period \( t \), \( \sum_{i} x_{i,t} \geq d_t \).
    - **Spare-Capacity Buffer:** For each period \( t \), \( \sum_{i} x_{i,t} \leq 0.9 \cdot \sum_{i} Q_i \cdot y_{i,t} \).
    - **Truck Capacity:** For all \( i, t \), \( 0 \leq x_{i,t} \leq Q_i \cdot y_{i,t} \).
    - **Startup Definition:** For all \( i, t \), \( s_{i,t} \geq y_{i,t} - y_{i,t-1} \), with \( y_{i,0} = 0 \) (all trucks initially off).
    - **No Startup in Last Period:** For all \( i \), \( s_{i,4} = 0 \).
    - **Minimum Up-Time:** If \( s_{i,t} = 1 \), then \( y_{i,t+1} = 1 \) for \( t = 1,2,3 \) (cannot start in period 4).
    - **Minimum Down-Time:** If truck \( i \) is shut down in period \( t \) (i.e., \( y_{i,t-1} = 1, y_{i,t} = 0 \)), then \( y_{i,t+1} = 0 \) and \( y_{i,t+2} = 0 \) (for \( t = 1,2 \)), and truck cannot be restarted before \( t+2 \).
    - **Ramp (Change) Constraints:** For all \( i, t = 2,3,4 \), \( |x_{i,t} - x_{i,t-1}| \leq 300 \).
    - **Inactive Truck Zero Load:** For all \( i, t \), \( x_{i,t} = 0 \) if \( y_{i,t} = 0 \).
[Abstract Model Plan END]