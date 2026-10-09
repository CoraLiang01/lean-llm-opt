[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the nonnegative integer number of units of each of 101 radio models to produce per day, so as to minimize the total idle production time across three workstations, given model-specific processing times and workstation-specific effective daily capacities (after maintenance).
2.  **Identify Model Type:** Based on the query, this is a Mixed Integer Programming (MIP) problem (integer variables, linear objective and constraints).
3.  **Define Index Sets:** The primary indices are:
    -   Radio models: \( i \in \{1, 2, ..., 101\} \) (corresponding to HiFi-1 to HiFi-101)
    -   Workstations: \( w \in \{1, 2, 3\} \)
4.  **Define Decision Variables:**
    -   \( x_i \) = Number of units of radio model \( i \) to produce per day. Type: GRB.INTEGER (nonnegative).
    -   \( \text{Idle}_w \) = Idle time (in minutes) at workstation \( w \) per day. Type: GRB.CONTINUOUS (nonnegative).
5.  **Identify Parameters (from Schema):**
    -   Processing time per unit of model \( i \) at workstation \( w \): from columns 'HiFi1_Minutes', ..., 'HiFi101_Minutes' in each row (workstation) of workstation_times.csv.
    -   Maintenance percentage for each workstation: from 'Maintenance_Percent' column.
    -   Total available time per workstation per day: 1,440 minutes (before maintenance).
    -   Effective daily capacity per workstation: \( 1,440 \times (1 - \text{Maintenance\_Percent}_w / 100) \).
6.  **Formulate Objective:** Minimize the total idle production time across all workstations, i.e., minimize \( \sum_{w=1}^3 \text{Idle}_w \), where for each workstation, idle time is its effective daily capacity minus the total processing time used for all produced units.
7.  **Formulate Constraints:**
    -   For each workstation \( w \):
        -   The total processing time used at workstation \( w \) is \( \sum_{i=1}^{101} \text{ProcessingTime}_{w,i} \times x_i \).
        -   Idle time at workstation \( w \) is defined as: \( \text{Idle}_w = \text{EffectiveCapacity}_w - \sum_{i=1}^{101} \text{ProcessingTime}_{w,i} \times x_i \).
        -   Idle time must be nonnegative: \( \text{Idle}_w \geq 0 \).
        -   Total processing time used cannot exceed effective capacity: \( \sum_{i=1}^{101} \text{ProcessingTime}_{w,i} \times x_i \leq \text{EffectiveCapacity}_w \).
    -   Nonnegativity and integrality: \( x_i \geq 0 \), integer for all \( i \); \( \text{Idle}_w \geq 0 \), continuous for all \( w \).
[Abstract Model Plan END]