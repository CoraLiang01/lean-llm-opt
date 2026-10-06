##### Variables

Let $w_j$ denote the daily wage (in yuan) for worker $j$, for all workers $j$ listed as columns in the file, in the given order.

##### Parameters

Let $D_{ij}$ denote the number of days worker $j$ spent renovating homeowner $i$'s home, as given in the CSV data. The set of workers is $W = \{$Carpenter, Electrician, Painter, Worker_004, ..., Worker_150$\}$, and the set of homeowners is $H = W$ (each worker is also a homeowner).

Let $N$ be the total number of workers (here, $N=150$).

##### Mathematical Model

###### 1. Wage Balance Constraints

For each participant $i \in H$ (i.e., for each homeowner/worker), the total income they receive from working on others’ homes equals their total expenditure for work performed at their own home:

$$
\sum_{\substack{k \in W \\ k \ne i}} D_{ki} \cdot w_i = \sum_{\substack{j \in W \\ j \ne i}} D_{ij} \cdot w_j \qquad \forall i \in W
$$

That is, for each $i$:
- The left side is the total income of $i$ (number of days $i$ worked on others’ homes, times their own wage).
- The right side is the total amount $i$ pays to others for work done at their own home.

###### 2. Wage Normalization Constraint

The daily wage of the first worker (Carpenter) is fixed:

$$
w_{\text{Carpenter}} = 60.00
$$

###### 3. Workday Totals (from problem statement)

Each worker contributes exactly 10 work days in total:

$$
\sum_{i \in H} D_{ij} = 10 \qquad \forall j \in W
$$

(This is a parameter property, not a constraint on variables, but included for completeness.)

###### 4. Non-negativity

$$
w_j \geq 0 \qquad \forall j \in W
$$

##### Retrieved Information

- Workers (in order): Carpenter, Electrician, Painter, Worker_004, ..., Worker_150
- Homeowners: Same as workers
- $D_{ij}$: Number of days worker $j$ spent on homeowner $i$'s home, as given in the CSV data (see above for full matrix).

##### Full Model (in summary)

Find $w_j$ for all $j \in W$ such that:

For all $i \in W$:
$$
\sum_{\substack{k \in W \\ k \ne i}} D_{ki} \cdot w_i = \sum_{\substack{j \in W \\ j \ne i}} D_{ij} \cdot w_j
$$

With:
$$
w_{\text{Carpenter}} = 60.00
$$

and
$$
w_j \geq 0 \quad \forall j \in W
$$

where $D_{ij}$ is as retrieved from the CSV data above.