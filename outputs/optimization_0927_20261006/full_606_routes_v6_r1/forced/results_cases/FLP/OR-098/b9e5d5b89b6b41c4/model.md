##### Parameters

Let $N$ be the number of participants (workers/homeowners), indexed by $i=1,\ldots,N$.
Let $W = \{$"Carpenter", "Electrician", "Painter", "Worker_004", ..., "Worker_150"$\}$ be the ordered list of workers (and homeowners).
Let $a_{ij}$ denote the number of days worker $j$ worked on homeowner $i$'s home, as given in the CSV (row $i$, column $j$).
Let $w_j$ denote the daily wage of worker $j$.

##### Decision Variables

$w_j \geq 0$: daily wage of worker $j$, for $j=1,\ldots,N$.

##### Fixed Parameter

$w_1 = 60.00$ (the daily wage of the first worker, "Carpenter", is fixed at 60.00 yuan).

##### Constraints

1. **Work Contribution Constraint**:  
   Each worker contributes exactly 10 days in total:
   $$
   \sum_{i=1}^N a_{ij} = 10, \quad \forall j=1,\ldots,N
   $$
   (This is satisfied by the data, so not enforced as a constraint in the model.)

2. **Mutual Payment Balance for Each Participant**:  
   For each participant $i=1,\ldots,N$ (homeowner $i$), the total income they receive from working on others' homes equals their total expenditure for work performed at their own home:
   $$
   \sum_{\substack{j=1 \\ j \neq i}}^N a_{ji} w_i = \sum_{\substack{j=1 \\ j \neq i}}^N a_{ij} w_j, \quad \forall i=1,\ldots,N
   $$
   That is, for each $i$:
   $$
   \left(\sum_{j \neq i} a_{ji}\right) w_i - \sum_{j \neq i} a_{ij} w_j = 0
   $$

   Or, equivalently, for all $i=1,\ldots,N$:
   $$
   \sum_{j=1}^N a_{ji} w_i - \sum_{j=1}^N a_{ij} w_j = 0
   $$
   (since $a_{ii} w_i$ cancels on both sides).

3. **Wage Fixing**:  
   $$
   w_1 = 60.00
   $$

##### Variable Domains

$w_j \geq 0$ for all $j=1,\ldots,N$.

##### Data

- $a_{ij}$: as given in the full CSV, with $i$ indexing rows ("Owner") and $j$ indexing columns (worker names).
- $N = 150$ (from the data).
- $W = $ ["Carpenter", "Electrician", "Painter", "Worker_004", ..., "Worker_150"] (full list as in the CSV header).

##### Model Summary

Find $w_j$ for $j=1,\ldots,N$ such that:
- $w_1 = 60.00$
- For all $i=1,\ldots,N$:
  $$
  \sum_{j=1}^N a_{ji} w_i - \sum_{j=1}^N a_{ij} w_j = 0
  $$
- $w_j \geq 0$ for all $j$.

##### Retrieved Data

- $a_{ij}$: full $150 \times 150$ matrix from work_days.csv, with row and column identifiers preserved.
- $W$: ["Carpenter", "Electrician", "Painter", "Worker_004", ..., "Worker_150"] (in CSV order).
- $N = 150$.

##### Notes

- The system is a square linear system in $w_j$ (with $w_1$ fixed).
- All identifiers and coefficients are as in the CSV.
- No objective function is needed; the solution is determined by the balance and normalization constraints.