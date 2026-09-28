##### Sets and Indices

Let $M = \{1,2,\ldots,101\}$ index the radio models (HiFi-1, HiFi-2, ..., HiFi-101).

Let $W = \{1,2,3\}$ index the workstations.

##### Parameters

- $t_{w,m}$: processing time (in minutes) required at workstation $w$ for one unit of model $m$, from the CSV below.
- $C_w$: total daily time available at workstation $w$ (all $=1440$ minutes).
- $p_w$: maintenance percentage at workstation $w$ ($p_1=0.10$, $p_2=0.14$, $p_3=0.12$).
- $E_w = C_w \cdot (1-p_w)$: effective daily capacity at workstation $w$.

##### Decision Variables

- $x_m \in \mathbb{Z}_+, \quad \forall m \in M$: number of units of model $m$ to produce per day (nonnegative integer).

##### Objective Function

Minimize total idle production time across all workstations:
$$
\min \sum_{w=1}^3 \left[ E_w - \sum_{m=1}^{101} t_{w,m} x_m \right]
$$

##### Constraints

1. Cannot use more than effective capacity at any workstation:
$$
\sum_{m=1}^{101} t_{w,m} x_m \leq E_w, \quad \forall w \in \{1,2,3\}
$$

2. Nonnegativity and integrality:
$$
x_m \in \mathbb{Z}_+, \quad \forall m \in M
$$

##### Data

- $C_1 = C_2 = C_3 = 1440$
- $p_1 = 0.10$, $p_2 = 0.14$, $p_3 = 0.12$
- $E_1 = 1440 \times 0.90 = 1296$
- $E_2 = 1440 \times 0.86 = 1238.4$
- $E_3 = 1440 \times 0.88 = 1267.2$

- $t_{w,m}$: as given in the CSV below (source order preserved):

| Workstation | HiFi1 | HiFi2 | HiFi3 | ... | HiFi101 | Maintenance_Percent |
|-------------|-------|-------|-------|-----|---------|---------------------|
| 1           | 6     | 4     | 6     | ... | 10      | 10                  |
| 2           | 5     | 5     | 5     | ... | 3       | 14                  |
| 3           | 4     | 6     | 5     | ... | 6       | 12                  |

(Full $t_{w,m}$ matrix as in the retrieved CSV.)

##### Complete Numerical Formulation

Let $x_m$ be the integer number of units of HiFi-$m$ to produce per day, $m=1,\ldots,101$.

Minimize
$$
\left[1296 - \sum_{m=1}^{101} t_{1,m} x_m\right] + \left[1238.4 - \sum_{m=1}^{101} t_{2,m} x_m\right] + \left[1267.2 - \sum_{m=1}^{101} t_{3,m} x_m\right]
$$

Subject to:
$$
\sum_{m=1}^{101} t_{1,m} x_m \leq 1296
$$
$$
\sum_{m=1}^{101} t_{2,m} x_m \leq 1238.4
$$
$$
\sum_{m=1}^{101} t_{3,m} x_m \leq 1267.2
$$
$$
x_m \in \mathbb{Z}_+, \quad m=1,\ldots,101
$$

Where $t_{w,m}$ are the coefficients from the CSV:

- For $w=1$, $t_{1,1}=6$, $t_{1,2}=4$, ..., $t_{1,101}=10$
- For $w=2$, $t_{2,1}=5$, $t_{2,2}=5$, ..., $t_{2,101}=3$
- For $w=3$, $t_{3,1}=4$, $t_{3,2}=6$, ..., $t_{3,101}=6$

All coefficients and identifiers are preserved as in the source data.