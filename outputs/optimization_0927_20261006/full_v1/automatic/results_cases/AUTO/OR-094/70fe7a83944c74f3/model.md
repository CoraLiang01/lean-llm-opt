Let $x_k$ be the number of units of radio model HiFi$k$ ($k=1,\ldots,101$) to produce per day. All $x_k \in \mathbb{Z}_{\geq 0}$.

Let $t_{jk}$ be the processing time (in minutes) required for one unit of model $k$ at workstation $j$ ($j=1,2,3$).

Let $C_j$ be the effective daily capacity (in minutes) of workstation $j$ after maintenance.

Define the idle time at workstation $j$ as $I_j = C_j - \sum_{k=1}^{101} t_{jk} x_k$.

Minimize total idle time:
\[
\min \sum_{j=1}^3 I_j = \sum_{j=1}^3 \left( C_j - \sum_{k=1}^{101} t_{jk} x_k \right)
\]
which is equivalent to:
\[
\max \sum_{j=1}^3 \sum_{k=1}^{101} t_{jk} x_k
\]
subject to the constraints below.

#### Data

- Workstation 1: Maintenance_Percent = 10, so $C_1 = 1440 \times (1 - 0.10) = 1296$ minutes
- Workstation 2: Maintenance_Percent = 14, so $C_2 = 1440 \times (1 - 0.14) = 1238.4$ minutes
- Workstation 3: Maintenance_Percent = 12, so $C_3 = 1440 \times (1 - 0.12) = 1267.2$ minutes

- For $j=1$ (Workstation 1), $t_{1,k}$ is as follows (in source order):

| $k$ | Model         | $t_{1k}$ |
|-----|--------------|----------|
| 1   | HiFi1        | 6        |
| 2   | HiFi2        | 4        |
| 3   | HiFi3        | 6        |
| 4   | HiFi4        | 7        |
| 5   | HiFi5        | 6        |
| 6   | HiFi6        | 6        |
| 7   | HiFi7        | 8        |
| 8   | HiFi8        | 9        |
| 9   | HiFi9        | 6        |
| 10  | HiFi10       | 7        |
| 11  | HiFi11       | 1        |
| 12  | HiFi12       | 2        |
| 13  | HiFi13       | 4        |
| 14  | HiFi14       | 7        |
| 15  | HiFi15       | 3        |
| 16  | HiFi16       | 8        |
| 17  | HiFi17       | 3        |
| 18  | HiFi18       | 2        |
| 19  | HiFi19       | 4        |
| 20  | HiFi20       | 5        |
| 21  | HiFi21       | 8        |
| 22  | HiFi22       | 3        |
| 23  | HiFi23       | 2        |
| 24  | HiFi24       | 3        |
| 25  | HiFi25       | 9        |
| 26  | HiFi26       | 7        |
| 27  | HiFi27       | 3        |
| 28  | HiFi28       | 5        |
| 29  | HiFi29       | 7        |
| 30  | HiFi30       | 6        |
| 31  | HiFi31       | 2        |
| 32  | HiFi32       | 1        |
| 33  | HiFi33       | 5        |
| 34  | HiFi34       | 6        |
| 35  | HiFi35       | 5        |
| 36  | HiFi36       | 1        |
| 37  | HiFi37       | 7        |
| 38  | HiFi38       | 9        |
| 39  | HiFi39       | 8        |
| 40  | HiFi40       | 3        |
| 41  | HiFi41       | 3        |
| 42  | HiFi42       | 8        |
| 43  | HiFi43       | 2        |
| 44  | HiFi44       | 3        |
| 45  | HiFi45       | 3        |
| 46  | HiFi46       | 8        |
| 47  | HiFi47       | 9        |
| 48  | HiFi48       | 2        |
| 49  | HiFi49       | 3        |
| 50  | HiFi50       | 4        |
| 51  | HiFi51       | 2        |
| 52  | HiFi52       | 9        |
| 53  | HiFi53       | 2        |
| 54  | HiFi54       | 1        |
| 55  | HiFi55       | 8        |
| 56  | HiFi56       | 8        |
| 57  | HiFi57       | 4        |
| 58  | HiFi58       | 4        |
| 59  | HiFi59       | 6        |
| 60  | HiFi60       | 1        |
| 61  | HiFi61       | 6        |
| 62  | HiFi62       | 5        |
| 63  | HiFi63       | 3        |
| 64  | HiFi64       | 5        |
| 65  | HiFi65       | 1        |
| 66  | HiFi66       | 6        |
| 67  | HiFi67       | 6        |
| 68  | HiFi68       | 5        |
| 69  | HiFi69       | 3        |
| 70  | HiFi70       | 4        |
| 71  | HiFi71       | 3        |
| 72  | HiFi72       | 8        |
| 73  | HiFi73       | 1        |
| 74  | HiFi74       | 2        |
| 75  | HiFi75       | 3        |
| 76  | HiFi76       | 2        |
| 77  | HiFi77       | 8        |
| 78  | HiFi78       | 4        |
| 79  | HiFi79       | 4        |
| 80  | HiFi80       | 2        |
| 81  | HiFi81       | 7        |
| 82  | HiFi82       | 5        |
| 83  | HiFi83       | 1        |
| 84  | HiFi84       | 6        |
| 85  | HiFi85       | 4        |
| 86  | HiFi86       | 1        |
| 87  | HiFi87       | 3        |
| 88  | HiFi88       | 8        |
| 89  | HiFi89       | 3        |
| 90  | HiFi90       | 3        |
| 91  | HiFi91       | 3        |
| 92  | HiFi92       | 3        |
| 93  | HiFi93       | 6        |
| 94  | HiFi94       | 7        |
| 95  | HiFi95       | 6        |
| 96  | HiFi96       | 2        |
| 97  | HiFi97       | 1        |
| 98  | HiFi98       | 8        |
| 99  | HiFi99       | 9        |
| 100 | HiFi100      | 7        |
| 101 | HiFi101      | 9        |

- For $j=2$ (Workstation 2), $t_{2k}$ is as follows (in source order):

| $k$ | Model         | $t_{2k}$ |
|-----|--------------|----------|
| 1   | HiFi1        | 5        |
| 2   | HiFi2        | 5        |
| 3   | HiFi3        | 5        |
| 4   | HiFi4        | 1        |
| 5   | HiFi5        | 7        |
| 6   | HiFi6        | 8        |
| 7   | HiFi7        | 7        |
| 8   | HiFi8        | 5        |
| 9   | HiFi9        | 6        |
| 10  | HiFi10       | 8        |
| 11  | HiFi11       | 9        |
| 12  | HiFi12       | 9        |
| 13  | HiFi13       | 2        |
| 14  | HiFi14       | 6        |
| 15  | HiFi15       | 9        |
| 16  | HiFi16       | 4        |
| 17  | HiFi17       | 1        |
| 18  | HiFi18       | 2        |
| 19  | HiFi19       | 9        |
| 20  | HiFi20       | 3        |
| 21  | HiFi21       | 8        |
| 22  | HiFi22       | 5        |
| 23  | HiFi23       | 9        |
| 24  | HiFi24       | 5        |
| 25  | HiFi25       | 8        |
| 26  | HiFi26       | 7        |
| 27  | HiFi27       | 1        |
| 28  | HiFi28       | 1        |
| 29  | HiFi29       | 9        |
| 30  | HiFi30       | 7        |
| 31  | HiFi31       | 1        |
| 32  | HiFi32       | 9        |
| 33  | HiFi33       | 6        |
| 34  | HiFi34       | 4        |
| 35  | HiFi35       | 7        |
| 36  | HiFi36       | 4        |
| 37  | HiFi37       | 8        |
| 38  | HiFi38       | 6        |
| 39  | HiFi39       | 5        |
| 40  | HiFi40       | 3        |
| 41  | HiFi41       | 6        |
| 42  | HiFi42       | 7        |
| 43  | HiFi43       | 6        |
| 44  | HiFi44       | 2        |
| 45  | HiFi45       | 1        |
| 46  | HiFi46       | 1        |
| 47  | HiFi47       | 3        |
| 48  | HiFi48       | 8        |
| 49  | HiFi49       | 4        |
| 50  | HiFi50       | 3        |
| 51  | HiFi51       | 6        |
| 52  | HiFi52       | 9        |
| 53  | HiFi53       | 8        |
| 54  | HiFi54       | 7        |
| 55  | HiFi55       | 2        |
| 56  | HiFi56       | 2        |
| 57  | HiFi57       | 5        |
| 58  | HiFi58       | 4        |
| 59  | HiFi59       | 3        |
| 60  | HiFi60       | 8        |
| 61  | HiFi61       | 8        |
| 62  | HiFi62       | 6        |
| 63  | HiFi63       | 6        |
| 64  | HiFi64       | 3        |
| 65  | HiFi65       | 1        |
| 66  | HiFi66       | 6        |
| 67  | HiFi67       | 2        |
| 68  | HiFi68       | 6        |
| 69  | HiFi69       | 1        |
| 70  | HiFi70       | 3        |
| 71  | HiFi71       | 7        |
| 72  | HiFi72       | 1        |
| 73  | HiFi73       | 1        |
| 74  | HiFi74       | 2        |
| 75  | HiFi75       | 8        |
| 76  | HiFi76       | 7        |
| 77  | HiFi77       | 8        |
| 78  | HiFi78       | 8        |
| 79  | HiFi79       | 7        |
| 80  | HiFi80       | 5        |
| 81  | HiFi81       | 2        |
| 82  | HiFi82       | 5        |
| 83  | HiFi83       | 6        |
| 84  | HiFi84       | 2        |
| 85  | HiFi85       | 3        |
| 86  | HiFi86       | 2        |
| 87  | HiFi87       | 3        |
| 88  | HiFi88       | 8        |
| 89  | HiFi89       | 4        |
| 90  | HiFi90       | 9        |
| 91  | HiFi91       | 6        |
| 92  | HiFi92       | 1        |
| 93  | HiFi93       | 4        |
| 94  | HiFi94       | 8        |
| 95  | HiFi95       | 8        |
| 96  | HiFi96       | 6        |
| 97  | HiFi97       | 8        |
| 98  | HiFi98       | 5        |
| 99  | HiFi99       | 5        |
| 100 | HiFi100      | 8        |
| 101 | HiFi101      | 3        |

- For $j=3$ (Workstation 3), $t_{3k}$ is as follows (in source order):

| $k$ | Model         | $t_{3k}$ |
|-----|--------------|----------|
| 1   | HiFi1        | 4        |
| 2   | HiFi2        | 6        |
| 3   | HiFi3        | 5        |
| 4   | HiFi4        | 2        |
| 5   | HiFi5        | 6        |
| 6   | HiFi6        | 5        |
| 7   | HiFi7        | 3        |
| 8   | HiFi8        | 3        |
| 9   | HiFi9        | 4        |
| 10  | HiFi10       | 8        |
| 11  | HiFi11       | 6        |
| 12  | HiFi12       | 3        |
| 13  | HiFi13       | 3        |
| 14  | HiFi14       | 3        |
| 15  | HiFi15       | 7        |
| 16  | HiFi16       | 8        |
| 17  | HiFi17       | 3        |
| 18  | HiFi18       | 8        |
| 19  | HiFi19       | 1        |
| 20  | HiFi20       | 5        |
| 21  | HiFi21       | 3        |
| 22  | HiFi22       | 8        |
| 23  | HiFi23       | 5        |
| 24  | HiFi24       | 8        |
| 25  | HiFi25       | 4        |
| 26  | HiFi26       | 8        |
| 27  | HiFi27       | 6        |
| 28  | HiFi28       | 7        |
| 29  | HiFi29       | 9        |
| 30  | HiFi30       | 5        |
| 31  | HiFi31       | 3        |
| 32  | HiFi32       | 6        |
| 33  | HiFi33       | 3        |
| 34  | HiFi34       | 3        |
| 35  | HiFi35       | 3        |
| 36  | HiFi36       | 8        |
| 37  | HiFi37       | 4        |
| 38  | HiFi38       | 6        |
| 39  | HiFi39       | 3        |
| 40  | HiFi40       | 8        |
| 41  | HiFi41       | 3        |
| 42  | HiFi42       | 7        |
| 43  | HiFi43       | 5        |
| 44  | HiFi44       | 3        |
| 45  | HiFi45       | 1        |
| 46  | HiFi46       | 8        |
| 47  | HiFi47       | 9        |
| 48  | HiFi48       | 6        |
| 49  | HiFi49       | 6        |
| 50  | HiFi50       | 4        |
| 51  | HiFi51       | 7        |
| 52  | HiFi52       | 1        |
| 53  | HiFi53       | 9        |
| 54  | HiFi54       | 9        |
| 55  | HiFi55       | 3        |
| 56  | HiFi56       | 9        |
| 57  | HiFi57       | 6        |
| 58  | HiFi58       | 5        |
| 59  | HiFi59       | 7        |
| 60  | HiFi60       | 8        |
| 61  | HiFi61       | 9        |
| 62  | HiFi62       | 9        |
| 63  | HiFi63       | 8        |
| 64  | HiFi64       | 5        |
| 65  | HiFi65       | 4        |
| 66  | HiFi66       | 4        |
| 67  | HiFi67       | 3        |
| 68  | HiFi68       | 3        |
| 69  | HiFi69       | 8        |
| 70  | HiFi70       | 8        |
| 71  | HiFi71       | 2        |
| 72  | HiFi72       | 4        |
| 73  | HiFi73       | 9        |
| 74  | HiFi74       | 6        |
| 75  | HiFi75       | 7        |
| 76  | HiFi76       | 6        |
| 77  | HiFi77       | 7        |
| 78  | HiFi78       | 3        |
| 79  | HiFi79       | 1        |
| 80  | HiFi80       | 7        |
| 81  | HiFi81       | 6        |
| 82  | HiFi82       | 4        |
| 83  | HiFi83       | 3        |
| 84  | HiFi84       | 5        |
| 85  | HiFi85       | 7        |
| 86  | HiFi86       | 6        |
| 87  | HiFi87       | 3        |
| 88  | HiFi88       | 5        |
| 89  | HiFi89       | 2        |
| 90  | HiFi90       | 2        |
| 91  | HiFi91       | 9        |
| 92  | HiFi92       | 3        |
| 93  | HiFi93       | 6        |
| 94  | HiFi94       | 9        |
| 95  | HiFi95       | 7        |
| 96  | HiFi96       | 2        |
| 97  | HiFi97       | 4        |
| 98  | HiFi98       | 5        |
| 99  | HiFi99       | 8        |
| 100 | HiFi100      | 1        |
| 101 | HiFi101      | 6        |

#### Mathematical Model

Minimize total idle time:
\[
\min \left[ (1296 - \sum_{k=1}^{101} t_{1k} x_k) + (1238.4 - \sum_{k=1}^{101} t_{2k} x_k) + (1267.2 - \sum_{k=1}^{101} t_{3k} x_k) \right]
\]
which is equivalent to:
\[
\max \sum_{j=1}^3 \sum_{k=1}^{101} t_{jk} x_k
\]
subject to:

\[
\sum_{k=1}^{101} t_{1k} x_k \leq 1296
\]
\[
\sum_{k=1}^{101} t_{2k} x_k \leq 1238.4
\]
\[
\sum_{k=1}^{101} t_{3k} x_k \leq 1267.2
\]
\[
x_k \in \mathbb{Z}_{\geq 0} \quad \forall k=1,\ldots,101
\]

where all $t_{jk}$ are as given above, in the original source order.