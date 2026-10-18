[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the non‑negative integer production quantities for 101 radio models (HiFi‑1 … HiFi‑101) on three workstations so that the total idle time across the workstations is minimized, given the per‑unit processing times and the effective daily capacities after maintenance.

2.  **Identify Model Type:** This is a pure Integer Linear Programming (ILP) problem (a special case of MIP with only integer variables).

3.  **Define Index Sets:**  
   - `Models = {1, 2, …, 101}` (HiFi‑1 … HiFi‑101)  
   - `Workstations = {1, 2, 3}`

4.  **Define Decision Variables:**  
   - `x[i]` = number of units of model *i* produced per day.  
     Type: `GRB.INTEGER` (non‑negative).  
   - No binary or continuous auxiliary variables are required.

5.  **Identify Parameters (from Schema):**  
   - `t[w][i]` = processing time (minutes) required for one unit of model *i* at workstation *w* (from columns `HiFi{i}_Minutes`).  
   - `Maintenance_Percent[w]` = maintenance percentage for workstation *w* (from column `Maintenance_Percent`).  
   - `EffectiveCapacity[w]` = 1440 × (1 – Maintenance_Percent[w]/100) (effective available minutes per day).  
   - All parameters are constants derived directly from the CSV file.

6.  **Formulate Objective:**  
   Minimize total idle time:  
   \[
   \min \sum_{w \in Workstations}\Bigl(\text{EffectiveCapacity}[w] - \sum_{i \in Models} t[w][i] \cdot x[i]\Bigr)
   \]
   (Since the capacities are constants, this is equivalent to maximizing the total used processing time.)

7.  **Formulate Constraints:**  
   - **Workstation capacity limits** (for each workstation *w*):  
     \[
     \sum_{i \in Models} t[w][i] \cdot x[i] \;\le\; \text{EffectiveCapacity}[w]
     \]
   - **Non‑negativity and integrality** (for each model *i*):  
     \[
     x[i] \;\ge\; 0,\quad x[i] \in \mathbb{Z}
     \]
[Abstract Model Plan END]