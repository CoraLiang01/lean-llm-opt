Let \( x_j \) be the nonnegative integer number of units of radio model \( j \) (for \( j = 1, \ldots, 101 \), corresponding to HiFi1, HiFi2, ..., HiFi101) to produce per day.

Let \( t_{i,j} \) be the processing time (in minutes) required per unit of model \( j \) at workstation \( i \), as given in the CSV data below.

Let the effective daily capacity (in minutes) of each workstation be:
- Workstation 1: \( 1440 \times (1 - 0.10) = 1296 \) minutes
- Workstation 2: \( 1440 \times (1 - 0.14) = 1238.4 \) minutes
- Workstation 3: \( 1440 \times (1 - 0.12) = 1267.2 \) minutes

Define the idle time at workstation \( i \) as:
\[
\text{Idle}_i = \text{EffectiveCapacity}_i - \sum_{j=1}^{101} t_{i,j} x_j
\]
The objective is to minimize the total idle time across all workstations:
\[
\min \sum_{i=1}^3 \text{Idle}_i = \sum_{i=1}^3 \left( \text{EffectiveCapacity}_i - \sum_{j=1}^{101} t_{i,j} x_j \right)
\]
which is equivalent to:
\[
\min \left( \sum_{i=1}^3 \text{EffectiveCapacity}_i - \sum_{i=1}^3 \sum_{j=1}^{101} t_{i,j} x_j \right)
\]
Since the effective capacities are constant, this is equivalent to maximizing total processing time used:
\[
\max \sum_{i=1}^3 \sum_{j=1}^{101} t_{i,j} x_j
\]
subject to the capacity constraints at each workstation.

Full integer programming formulation:

**Variables:**
- \( x_j \in \mathbb{Z}_+, \quad j = 1, \ldots, 101 \) (number of units of each model to produce per day)

**Parameters (from CSV):**
- \( t_{i,j} \): processing time (in minutes) per unit of model \( j \) at workstation \( i \), as given below.

**Objective:**
\[
\min \left[ (1296 + 1238.4 + 1267.2) - \sum_{i=1}^3 \sum_{j=1}^{101} t_{i,j} x_j \right]
\]
or equivalently,
\[
\max \sum_{i=1}^3 \sum_{j=1}^{101} t_{i,j} x_j
\]

**Subject to:**
\[
\sum_{j=1}^{101} t_{1,j} x_j \leq 1296
\]
\[
\sum_{j=1}^{101} t_{2,j} x_j \leq 1238.4
\]
\[
\sum_{j=1}^{101} t_{3,j} x_j \leq 1267.2
\]
\[
x_j \in \mathbb{Z}_+, \quad \forall j = 1, \ldots, 101
\]

**Data (from workstation_times.csv):**

Let the models be indexed as follows:
- \( j = 1 \) is HiFi1, \( j = 2 \) is HiFi2, ..., \( j = 101 \) is HiFi101.

The processing times \( t_{i,j} \) are as follows (rows: workstations 1, 2, 3; columns: HiFi1_Minutes, ..., HiFi101_Minutes):

Workstation 1:  
t_{1,1}=6, t_{1,2}=4, t_{1,3}=6, t_{1,4}=7, t_{1,5}=6, t_{1,6}=6, t_{1,7}=8, t_{1,8}=9, t_{1,9}=6, t_{1,10}=7, t_{1,11}=1, t_{1,12}=2, t_{1,13}=4, t_{1,14}=7, t_{1,15}=3, t_{1,16}=8, t_{1,17}=3, t_{1,18}=2, t_{1,19}=4, t_{1,20}=5, t_{1,21}=8, t_{1,22}=3, t_{1,23}=2, t_{1,24}=3, t_{1,25}=9, t_{1,26}=7, t_{1,27}=3, t_{1,28}=5, t_{1,29}=7, t_{1,30}=6, t_{1,31}=2, t_{1,32}=1, t_{1,33}=5, t_{1,34}=6, t_{1,35}=5, t_{1,36}=1, t_{1,37}=7, t_{1,38}=9, t_{1,39}=8, t_{1,40}=3, t_{1,41}=3, t_{1,42}=8, t_{1,43}=2, t_{1,44}=3, t_{1,45}=3, t_{1,46}=8, t_{1,47}=9, t_{1,48}=2, t_{1,49}=3, t_{1,50}=4, t_{1,51}=2, t_{1,52}=9, t_{1,53}=2, t_{1,54}=1, t_{1,55}=8, t_{1,56}=8, t_{1,57}=4, t_{1,58}=4, t_{1,59}=6, t_{1,60}=1, t_{1,61}=6, t_{1,62}=5, t_{1,63}=3, t_{1,64}=5, t_{1,65}=1, t_{1,66}=6, t_{1,67}=6, t_{1,68}=5, t_{1,69}=3, t_{1,70}=4, t_{1,71}=3, t_{1,72}=8, t_{1,73}=1, t_{1,74}=2, t_{1,75}=3, t_{1,76}=2, t_{1,77}=8, t_{1,78}=4, t_{1,79}=4, t_{1,80}=2, t_{1,81}=7, t_{1,82}=5, t_{1,83}=1, t_{1,84}=6, t_{1,85}=4, t_{1,86}=1, t_{1,87}=3, t_{1,88}=8, t_{1,89}=3, t_{1,90}=3, t_{1,91}=3, t_{1,92}=6, t_{1,93}=7, t_{1,94}=6, t_{1,95}=2, t_{1,96}=1, t_{1,97}=8, t_{1,98}=9, t_{1,99}=7, t_{1,100}=9, t_{1,101}=10

Workstation 2:  
t_{2,1}=5, t_{2,2}=5, t_{2,3}=5, t_{2,4}=1, t_{2,5}=7, t_{2,6}=8, t_{2,7}=7, t_{2,8}=5, t_{2,9}=6, t_{2,10}=8, t_{2,11}=9, t_{2,12}=9, t_{2,13}=2, t_{2,14}=6, t_{2,15}=9, t_{2,16}=4, t_{2,17}=1, t_{2,18}=2, t_{2,19}=9, t_{2,20}=3, t_{2,21}=8, t_{2,22}=5, t_{2,23}=9, t_{2,24}=5, t_{2,25}=8, t_{2,26}=7, t_{2,27}=1, t_{2,28}=1, t_{2,29}=9, t_{2,30}=7, t_{2,31}=1, t_{2,32}=9, t_{2,33}=6, t_{2,34}=4, t_{2,35}=7, t_{2,36}=4, t_{2,37}=8, t_{2,38}=6, t_{2,39}=5, t_{2,40}=3, t_{2,41}=6, t_{2,42}=7, t_{2,43}=6, t_{2,44}=2, t_{2,45}=1, t_{2,46}=1, t_{2,47}=3, t_{2,48}=8, t_{2,49}=4, t_{2,50}=3, t_{2,51}=6, t_{2,52}=9, t_{2,53}=8, t_{2,54}=7, t_{2,55}=2, t_{2,56}=2, t_{2,57}=5, t_{2,58}=4, t_{2,59}=3, t_{2,60}=8, t_{2,61}=8, t_{2,62}=6, t_{2,63}=6, t_{2,64}=3, t_{2,65}=1, t_{2,66}=6, t_{2,67}=2, t_{2,68}=6, t_{2,69}=1, t_{2,70}=3, t_{2,71}=7, t_{2,72}=1, t_{2,73}=1, t_{2,74}=2, t_{2,75}=8, t_{2,76}=7, t_{2,77}=8, t_{2,78}=8, t_{2,79}=7, t_{2,80}=5, t_{2,81}=2, t_{2,82}=5, t_{2,83}=6, t_{2,84}=2, t_{2,85}=3, t_{2,86}=2, t_{2,87}=3, t_{2,88}=8, t_{2,89}=4, t_{2,90}=9, t_{2,91}=6, t_{2,92}=1, t_{2,93}=4, t_{2,94}=8, t_{2,95}=8, t_{2,96}=6, t_{2,97}=8, t_{2,98}=5, t_{2,99}=5, t_{2,100}=8, t_{2,101}=3

Workstation 3:  
t_{3,1}=4, t_{3,2}=6, t_{3,3}=5, t_{3,4}=2, t_{3,5}=6, t_{3,6}=5, t_{3,7}=3, t_{3,8}=3, t_{3,9}=4, t_{3,10}=8, t_{3,11}=6, t_{3,12}=3, t_{3,13}=3, t_{3,14}=3, t_{3,15}=7, t_{3,16}=8, t_{3,17}=3, t_{3,18}=8, t_{3,19}=1, t_{3,20}=5, t_{3,21}=3, t_{3,22}=8, t_{3,23}=5, t_{3,24}=8, t_{3,25}=4, t_{3,26}=8, t_{3,27}=6, t_{3,28}=7, t_{3,29}=9, t_{3,30}=5, t_{3,31}=3, t_{3,32}=6, t_{3,33}=3, t_{3,34}=3, t_{3,35}=3, t_{3,36}=8, t_{3,37}=4, t_{3,38}=6, t_{3,39}=3, t_{3,40}=8, t_{3,41}=3, t_{3,42}=7, t_{3,43}=5, t_{3,44}=3, t_{3,45}=1, t_{3,46}=8, t_{3,47}=9, t_{3,48}=6, t_{3,49}=6, t_{3,50}=4, t_{3,51}=7, t_{3,52}=1, t_{3,53}=9, t_{3,54}=9, t_{3,55}=3, t_{3,56}=9, t_{3,57}=6, t_{3,58}=5, t_{3,59}=7, t_{3,60}=8, t_{3,61}=9, t_{3,62}=9, t_{3,63}=8, t_{3,64}=5, t_{3,65}=4, t_{3,66}=4, t_{3,67}=3, t_{3,68}=3, t_{3,69}=8, t_{3,70}=8, t_{3,71}=2, t_{3,72}=4, t_{3,73}=9, t_{3,74}=6, t_{3,75}=7, t_{3,76}=6, t_{3,77}=7, t_{3,78}=3, t_{3,79}=1, t_{3,80}=7, t_{3,81}=6, t_{3,82}=4, t_{3,83}=3, t_{3,84}=5, t_{3,85}=7, t_{3,86}=6, t_{3,87}=3, t_{3,88}=5, t_{3,89}=2, t_{3,90}=2, t_{3,91}=9, t_{3,92}=3, t_{3,93}=6, t_{3,94}=9, t_{3,95}=7, t_{3,96}=2, t_{3,97}=4, t_{3,98}=5, t_{3,99}=8, t_{3,100}=1, t_{3,101}=6

**Summary:**

Formulate the following integer program:

\[
\begin{align*}
\min_{x_1, \ldots, x_{101} \in \mathbb{Z}_+} \quad & (1296 + 1238.4 + 1267.2) - \sum_{i=1}^3 \sum_{j=1}^{101} t_{i,j} x_j \\
\text{s.t.} \quad & \sum_{j=1}^{101} t_{1,j} x_j \leq 1296 \\
& \sum_{j=1}^{101} t_{2,j} x_j \leq 1238.4 \\
& \sum_{j=1}^{101} t_{3,j} x_j \leq 1267.2 \\
& x_j \in \mathbb{Z}_+, \quad \forall j = 1, \ldots, 101
\end{align*}
\]
where all coefficients \( t_{i,j} \) are as listed above, and the variable \( x_j \) is the number of units of HiFi\( j \) to produce per day.