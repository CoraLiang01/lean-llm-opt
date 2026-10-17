Let $x_i$ denote the number of units to produce of component $i$ ($i = 1,2,\ldots,111$; component $i$ corresponds to $C_i$).

Let $p_i$ be the unit price of component $i$ (from unit_price.csv).

Let $a_{wi}$ be the unit processing time required for component $i$ in workshop $w$ (from processing_time_unit.csv, $w \in$ {Casting, Milling, Finishing, Assembly, QA & Packaging}).

Let $b_w$ be the total available working hours in workshop $w$ (from total_working_hours.csv).

The model is:

Maximize total output value:
$$
\max \sum_{i=1}^{111} p_i x_i
$$

Subject to workshop capacity constraints (for each workshop $w$):

$$
\sum_{i=1}^{111} a_{wi} x_i \leq b_w \qquad \forall w \in \{\text{Casting}, \text{Milling}, \text{Finishing}, \text{Assembly}, \text{QA \& Packaging}\}
$$

Nonnegativity and integrality:
$$
x_i \in \mathbb{Z}_{\geq 0} \qquad \forall i = 1,2,\ldots,111
$$

Where the data are:

**Component indices and unit prices (unit_price.csv, in source order):**

| $i$ | Component | $p_i$ |
|-----|-----------|-------|
| 1   | C1        | 193   |
| 2   | C2        | 64    |
| 3   | C3        | 103   |
| 4   | C4        | 210   |
| 5   | C5        | 85    |
| 6   | C6        | 126   |
| 7   | C7        | 226   |
| 8   | C8        | 94    |
| 9   | C9        | 73    |
| 10  | C10       | 120   |
| 11  | C11       | 81    |
| 12  | C12       | 94    |
| 13  | C13       | 133   |
| 14  | C14       | 197   |
| 15  | C15       | 63    |
| 16  | C16       | 159   |
| 17  | C17       | 160   |
| 18  | C18       | 97    |
| 19  | C19       | 182   |
| 20  | C20       | 128   |
| 21  | C21       | 181   |
| 22  | C22       | 171   |
| 23  | C23       | 91    |
| 24  | C24       | 228   |
| 25  | C25       | 152   |
| 26  | C26       | 85    |
| 27  | C27       | 203   |
| 28  | C28       | 134   |
| 29  | C29       | 232   |
| 30  | C30       | 125   |
| 31  | C31       | 181   |
| 32  | C32       | 246   |
| 33  | C33       | 226   |
| 34  | C34       | 88    |
| 35  | C35       | 187   |
| 36  | C36       | 152   |
| 37  | C37       | 130   |
| 38  | C38       | 86    |
| 39  | C39       | 50    |
| 40  | C40       | 229   |
| 41  | C41       | 93    |
| 42  | C42       | 169   |
| 43  | C43       | 72    |
| 44  | C44       | 67    |
| 45  | C45       | 136   |
| 46  | C46       | 118   |
| 47  | C47       | 101   |
| 48  | C48       | 94    |
| 49  | C49       | 78    |
| 50  | C50       | 76    |
| 51  | C51       | 155   |
| 52  | C52       | 114   |
| 53  | C53       | 225   |
| 54  | C54       | 238   |
| 55  | C55       | 59    |
| 56  | C56       | 135   |
| 57  | C57       | 245   |
| 58  | C58       | 231   |
| 59  | C59       | 219   |
| 60  | C60       | 167   |
| 61  | C61       | 164   |
| 62  | C62       | 139   |
| 63  | C63       | 220   |
| 64  | C64       | 167   |
| 65  | C65       | 240   |
| 66  | C66       | 170   |
| 67  | C67       | 91    |
| 68  | C68       | 106   |
| 69  | C69       | 135   |
| 70  | C70       | 91    |
| 71  | C71       | 51    |
| 72  | C72       | 73    |
| 73  | C73       | 211   |
| 74  | C74       | 189   |
| 75  | C75       | 169   |
| 76  | C76       | 153   |
| 77  | C77       | 151   |
| 78  | C78       | 167   |
| 79  | C79       | 173   |
| 80  | C80       | 152   |
| 81  | C81       | 101   |
| 82  | C82       | 216   |
| 83  | C83       | 196   |
| 84  | C84       | 92    |
| 85  | C85       | 92    |
| 86  | C86       | 97    |
| 87  | C87       | 224   |
| 88  | C88       | 128   |
| 89  | C89       | 139   |
| 90  | C90       | 109   |
| 91  | C91       | 206   |
| 92  | C92       | 161   |
| 93  | C93       | 227   |
| 94  | C94       | 187   |
| 95  | C95       | 106   |
| 96  | C96       | 248   |
| 97  | C97       | 82    |
| 98  | C98       | 222   |
| 99  | C99       | 209   |
| 100 | C100      | 223   |
| 101 | C101      | 204   |
| 102 | C102      | 114   |
| 103 | C103      | 146   |
| 104 | C104      | 231   |
| 105 | C105      | 93    |
| 106 | C106      | 224   |
| 107 | C107      | 220   |
| 108 | C108      | 100   |
| 109 | C109      | 187   |
| 110 | C110      | 213   |
| 111 | C111      | 142   |

**Workshops and total available hours (total_working_hours.csv):**

| Workshop         | $b_w$ |
|------------------|-------|
| Casting          | 7650  |
| Milling          | 6320  |
| Finishing        | 5538  |
| Assembly         | 5957  |
| QA & Packaging   | 6988  |

**Unit processing times $a_{wi}$ (processing_time_unit.csv, in source order):**

For each workshop $w$ and component $i$ ($C1$ to $C111$), $a_{wi}$ is the value in the corresponding row and column.

For example, for Casting:
- $a_{\text{Casting},1} = 0.74$, $a_{\text{Casting},2} = 0.77$, ..., $a_{\text{Casting},111} = 3.81$

For Milling:
- $a_{\text{Milling},1} = 0.6$, $a_{\text{Milling},2} = 3.38$, ..., $a_{\text{Milling},111} = 2.67$

For Finishing:
- $a_{\text{Finishing},1} = 0.0$, $a_{\text{Finishing},2} = 4.15$, ..., $a_{\text{Finishing},111} = 4.07$

For Assembly:
- $a_{\text{Assembly},1} = 4.84$, $a_{\text{Assembly},2} = 0.0$, ..., $a_{\text{Assembly},111} = 4.02$

For QA & Packaging:
- $a_{\text{QA \& Packaging},1} = 0.92$, $a_{\text{QA \& Packaging},2} = 0.0$, ..., $a_{\text{QA \& Packaging},111} = 3.21$

**Summary of the complete model:**

$$
\begin{align*}
\max\ & \sum_{i=1}^{111} p_i x_i \\
\text{s.t.}\quad
& \sum_{i=1}^{111} a_{\text{Casting},i} x_i \leq 7650 \\
& \sum_{i=1}^{111} a_{\text{Milling},i} x_i \leq 6320 \\
& \sum_{i=1}^{111} a_{\text{Finishing},i} x_i \leq 5538 \\
& \sum_{i=1}^{111} a_{\text{Assembly},i} x_i \leq 5957 \\
& \sum_{i=1}^{111} a_{\text{QA \& Packaging},i} x_i \leq 6988 \\
& x_i \in \mathbb{Z}_{\geq 0} \qquad \forall i = 1,\ldots,111
\end{align*}
$$

All coefficients and identifiers are as given in the retrieved data, in source order.