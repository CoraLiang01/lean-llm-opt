**Abstract Mathematical Model**

Let:

- $N$ = set of all workers (indexed by $n$), corresponding to all columns except "Owner" in `file_0_view_0` (i.e., ["Carpenter", "Electrician", "Painter", "Worker_004", ..., "Worker_150"])
- $H$ = set of all homeowners (indexed by $h$), corresponding to all rows in `file_0_view_0`, with each $h$ identified by the value in the "Owner" column
- $d_{h,n}$ = number of days worker $n$ worked on homeowner $h$'s home, from `file_0_view_0`, column $n$, row with "Owner" = $h$
- $w_n$ = daily wage of worker $n$ (decision variable, continuous, $\mathbb{R}_{\geq 0}$)

Given:

- Each worker $n$ contributed exactly 10 work days in total: $\sum_{h \in H} d_{h,n} = 10$
- The daily wage of the first worker (let's call this worker $n_0$) is fixed: $w_{n_0} = 60.00$

**Variables**

- $w_n \in \mathbb{R}_{\geq 0}$, $\forall n \in N$

**Constraints**

1. **Fair Mutual Payment for Each Participant:**

   For every participant $k \in N$ (i.e., for every homeowner $h$ with "Owner" = $k$):

   $$
   \sum_{\substack{n \in N \\ n \neq k}} d_{k,n} w_n = \sum_{\substack{h \in H \\ h \neq k}} d_{h,k} w_k
   $$

   - The left side is the total amount participant $k$ pays to others for work done at their own home.
   - The right side is the total income participant $k$ receives for working on others' homes.

   Since $w_k$ is a variable, and $d_{h,k}$ is the number of days $k$ worked on $h$'s home, this can be rewritten for all $k \in N$:

   $$
   \sum_{n \in N} d_{k,n} w_n - d_{k,k} w_k = \sum_{h \in H} d_{h,k} w_k - d_{k,k} w_k
   $$

   $$
   \sum_{n \in N} d_{k,n} w_n = \sum_{h \in H} d_{h,k} w_k
   $$

   But since $\sum_{h \in H} d_{h,k} = 10$ (from the data), this becomes:

   $$
   \sum_{n \in N} d_{k,n} w_n = 10 w_k, \quad \forall k \in N
   $$

2. **Wage Fixing:**

   $$
   w_{n_0} = 60.00
   $$

**Objective**

- No explicit objective is required; the system is determined by the constraints.

---

**Data Mapping**

- $N$ (workers): All columns except "Owner" in `file_0_view_0` (`work_days.csv`)
- $H$ (homeowners): All rows in `file_0_view_0`, with "Owner" as the identifier
- $d_{h,n}$: Value in `file_0_view_0`, row with "Owner" = $h$, column $n$
- $w_n$: Decision variable, daily wage for worker $n$
- $n_0$: The first worker column in `file_0_view_0` (i.e., the first column after "Owner")

---

**Summary of Model**

- **Variables:** $w_n \geq 0$ for all $n \in N$
- **Equations:** For all $k \in N$,
  $$
  \sum_{n \in N} d_{k,n} w_n = 10 w_k
  $$
- **Wage fixing:** $w_{n_0} = 60.00$
- **Data:** All $d_{h,n}$ from `file_0_view_0` (`work_days.csv`), with $h$ = "Owner" row, $n$ = worker column

---

**(End of Model)**