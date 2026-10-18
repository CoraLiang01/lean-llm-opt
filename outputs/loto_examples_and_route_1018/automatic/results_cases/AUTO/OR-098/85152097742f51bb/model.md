Let $N$ be the number of participants (workers/homeowners), and let the set of workers (in original column order) be $W = \{w_1, w_2, \ldots, w_N\}$, where $w_1$ is the first worker listed in the file (e.g., "Carpenter"). Let the set of homeowners (in original row order) be $O = \{o_1, o_2, \ldots, o_N\}$, where each $o_i$ corresponds to $w_i$ (i.e., $o_i$ is the home of worker $w_i$).

Let $d_{ij}$ denote the number of days worker $w_j$ spent renovating homeowner $o_i$'s home, as given in the retrieved work_days.csv file (preserving all original identifiers and order).

Let $r_j$ denote the daily wage of worker $w_j$ (decision variable for $j=2,\ldots,N$; $r_1$ is fixed).

The model is:

**Variables:**
- $r_j \in \mathbb{R}$, for $j=2,\ldots,N$ (daily wage of worker $w_j$)
- $r_1 = 60.00$ (daily wage of the first worker, e.g., Carpenter)

**Constraints:**

For each participant $i=1,\ldots,N$ (i.e., for each homeowner/worker):

\[
\sum_{\substack{k=1 \\ k \neq i}}^{N} d_{k i} \cdot r_i = \sum_{j=1}^{N} d_{i j} \cdot r_j
\]

That is, for each $i$:
- The left side is the total income worker $w_i$ earns from working on others' homes (sum over all $k \neq i$ of days $w_i$ worked on $o_k$'s home, times $r_i$).
- The right side is the total expenditure for work performed at $o_i$'s home (sum over all workers $w_j$ of days $w_j$ worked on $o_i$'s home, times $r_j$).

**Wage normalization:**
\[
r_1 = 60.00
\]

**Data:**

All $d_{ij}$ are taken directly from the retrieved work_days.csv, with $d_{ij}$ being the entry in row with Owner $o_i$ and column $w_j$.

**Complete Model:**

Given:
- Workers $w_1, w_2, \ldots, w_N$ (in original file order)
- Homeowners $o_1, o_2, \ldots, o_N$ (in original file order)
- $d_{ij}$: days worker $w_j$ spent on $o_i$'s home (from work_days.csv)

Find:
- $r_j$ for $j=2,\ldots,N$

Such that:
\[
\forall i=1,\ldots,N: \quad \left( \sum_{\substack{k=1 \\ k \neq i}}^{N} d_{k i} \right) r_i - \sum_{j=1}^{N} d_{i j} r_j = 0
\]
\[
r_1 = 60.00
\]

where all indices, names, and $d_{ij}$ values are as in the retrieved work_days.csv file, preserving original order and identifiers.