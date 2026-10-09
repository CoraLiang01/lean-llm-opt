**Sets and Indices:**
- $i \in \{1,2,3\}$: Workstation
- $k \in \{1,2,\ldots,101\}$: Radio model (HiFi1 to HiFi101)

**Parameters (from workstation_times.csv):**
- $a_{ik}$: Processing time (minutes) for one unit of HiFi-$k$ at workstation $i$  
  (from columns HiFi1_Minutes, ..., HiFi101_Minutes for each Workstation row)
- $C_1 = 1296$ (Workstation 1 effective minutes per day)
- $C_2 = 1238.4$ (Workstation 2 effective minutes per day)
- $C_3 = 1267.2$ (Workstation 3 effective minutes per day)

**Decision Variables:**
- $x_k \in \mathbb{Z}_{\geq 0}$: Number of units of HiFi-$k$ to produce per day

**Mathematical Model:**

Minimize total idle time:
$$
\min \sum_{i=1}^3 \left(C_i - \sum_{k=1}^{101} a_{ik} x_k\right)
$$

Equivalent to:
$$
\max \sum_{i=1}^3 \sum_{k=1}^{101} a_{ik} x_k
$$

Subject to:
\[
\sum_{k=1}^{101} a_{ik} x_k \leq C_i \qquad \forall i \in \{1,2,3\}
\]
\[
x_k \in \mathbb{Z}_{\geq 0} \qquad \forall k \in \{1,2,\ldots,101\}
\]

**Where:**

- For $i=1$ (Workstation 1, $C_1=1296$), $a_{1k}$ is the value in column HiFi$k$_Minutes, row Workstation=1.
- For $i=2$ (Workstation 2, $C_2=1238.4$), $a_{2k}$ is the value in column HiFi$k$_Minutes, row Workstation=2.
- For $i=3$ (Workstation 3, $C_3=1267.2$), $a_{3k}$ is the value in column HiFi$k$_Minutes, row Workstation=3.

**Explicitly, using the retrieved data:**

Let $x_k$ be the number of units of HiFi-$k$ produced per day, $k=1,\ldots,101$.

**Objective:**
\[
\min \left[
(1296 - \sum_{k=1}^{101} a_{1k} x_k) +
(1238.4 - \sum_{k=1}^{101} a_{2k} x_k) +
(1267.2 - \sum_{k=1}^{101} a_{3k} x_k)
\right]
\]

**Subject to:**
\[
\sum_{k=1}^{101} a_{1k} x_k \leq 1296
\]
\[
\sum_{k=1}^{101} a_{2k} x_k \leq 1238.4
\]
\[
\sum_{k=1}^{101} a_{3k} x_k \leq 1267.2
\]
\[
x_k \in \mathbb{Z}_{\geq 0} \quad \forall k=1,\ldots,101
\]

Where $a_{ik}$ are the coefficients from workstation_times.csv:

- For $i=1$: $a_{1k}$ = value in column HiFi$k$_Minutes, row Workstation=1
- For $i=2$: $a_{2k}$ = value in column HiFi$k$_Minutes, row Workstation=2
- For $i=3$: $a_{3k}$ = value in column HiFi$k$_Minutes, row Workstation=3

**All variables $x_k$ are nonnegative integers.**