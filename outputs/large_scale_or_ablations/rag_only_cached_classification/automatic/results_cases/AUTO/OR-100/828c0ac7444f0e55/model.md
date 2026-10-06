Let:
- \( x_j \) = number of units to produce of component \( j \), for \( j = 1, \ldots, 111 \), where \( j \) corresponds to component IDs C1, C2, ..., C111 in the order given.
- All \( x_j \) are nonnegative integers.

Parameters:
- Let \( p_j \) = unit price of component \( j \) (from unit_price.csv).
- Let \( a_{ij} \) = unit processing time (in hours) required for component \( j \) in workshop \( i \), where \( i \) is one of: Casting, Milling, Finishing, Assembly, QA & Packaging (from processing_time_unit.csv).
- Let \( b_i \) = total available working hours in workshop \( i \) (from total_working_hours.csv).

Indices:
- \( j \in \{1,2,\ldots,111\} \), corresponding to C1, C2, ..., C111.
- \( i \in \{\text{Casting}, \text{Milling}, \text{Finishing}, \text{Assembly}, \text{QA \& Packaging}\} \).

Data (in order):

unit_price.csv (component order C1 to C111):

\[
\begin{array}{ll}
\text{Component} & p_j \\
\text{C1} & 193 \\
\text{C2} & 64 \\
\text{C3} & 103 \\
\text{C4} & 210 \\
\text{C5} & 85 \\
\text{C6} & 126 \\
\text{C7} & 226 \\
\text{C8} & 94 \\
\text{C9} & 73 \\
\text{C10} & 120 \\
\text{C11} & 81 \\
\text{C12} & 94 \\
\text{C13} & 133 \\
\text{C14} & 197 \\
\text{C15} & 63 \\
\text{C16} & 159 \\
\text{C17} & 160 \\
\text{C18} & 97 \\
\text{C19} & 182 \\
\text{C20} & 128 \\
\text{C21} & 181 \\
\text{C22} & 171 \\
\text{C23} & 91 \\
\text{C24} & 228 \\
\text{C25} & 152 \\
\text{C26} & 85 \\
\text{C27} & 203 \\
\text{C28} & 134 \\
\text{C29} & 232 \\
\text{C30} & 125 \\
\text{C31} & 181 \\
\text{C32} & 246 \\
\text{C33} & 226 \\
\text{C34} & 88 \\
\text{C35} & 187 \\
\text{C36} & 152 \\
\text{C37} & 130 \\
\text{C38} & 86 \\
\text{C39} & 50 \\
\text{C40} & 229 \\
\text{C41} & 93 \\
\text{C42} & 169 \\
\text{C43} & 72 \\
\text{C44} & 67 \\
\text{C45} & 136 \\
\text{C46} & 118 \\
\text{C47} & 101 \\
\text{C48} & 94 \\
\text{C49} & 78 \\
\text{C50} & 76 \\
\text{C51} & 155 \\
\text{C52} & 114 \\
\text{C53} & 225 \\
\text{C54} & 238 \\
\text{C55} & 59 \\
\text{C56} & 135 \\
\text{C57} & 245 \\
\text{C58} & 231 \\
\text{C59} & 219 \\
\text{C60} & 167 \\
\text{C61} & 164 \\
\text{C62} & 139 \\
\text{C63} & 220 \\
\text{C64} & 167 \\
\text{C65} & 240 \\
\text{C66} & 170 \\
\text{C67} & 91 \\
\text{C68} & 106 \\
\text{C69} & 135 \\
\text{C70} & 91 \\
\text{C71} & 51 \\
\text{C72} & 73 \\
\text{C73} & 211 \\
\text{C74} & 189 \\
\text{C75} & 169 \\
\text{C76} & 153 \\
\text{C77} & 151 \\
\text{C78} & 167 \\
\text{C79} & 173 \\
\text{C80} & 152 \\
\text{C81} & 101 \\
\text{C82} & 216 \\
\text{C83} & 196 \\
\text{C84} & 92 \\
\text{C85} & 92 \\
\text{C86} & 97 \\
\text{C87} & 224 \\
\text{C88} & 128 \\
\text{C89} & 139 \\
\text{C90} & 109 \\
\text{C91} & 206 \\
\text{C92} & 161 \\
\text{C93} & 227 \\
\text{C94} & 187 \\
\text{C95} & 106 \\
\text{C96} & 248 \\
\text{C97} & 82 \\
\text{C98} & 222 \\
\text{C99} & 209 \\
\text{C100} & 223 \\
\text{C101} & 204 \\
\text{C102} & 114 \\
\text{C103} & 146 \\
\text{C104} & 231 \\
\text{C105} & 93 \\
\text{C106} & 224 \\
\text{C107} & 220 \\
\text{C108} & 100 \\
\text{C109} & 187 \\
\text{C110} & 213 \\
\text{C111} & 142 \\
\end{array}
\]

processing_time_unit.csv (workshop × component, in order):

Let \( a_{ij} \) be the value in row \( i \) (workshop), column \( j \) (component C1 to C111), as given in the CSV. For example, for Casting and C1, \( a_{\text{Casting},1} = 0.74 \), for Milling and C1, \( a_{\text{Milling},1} = 0.6 \), etc.

total_working_hours.csv:

\[
\begin{array}{ll}
\text{Workshop} & b_i \\
\text{Casting} & 7650 \\
\text{Milling} & 6320 \\
\text{Finishing} & 5538 \\
\text{Assembly} & 5957 \\
\text{QA \& Packaging} & 6988 \\
\end{array}
\]

Model:

\[
\begin{align*}
\text{Maximize} \quad & Z = \sum_{j=1}^{111} p_j x_j \\
\text{subject to} \quad
& \sum_{j=1}^{111} a_{\text{Casting},j} x_j \leq 7650 \\
& \sum_{j=1}^{111} a_{\text{Milling},j} x_j \leq 6320 \\
& \sum_{j=1}^{111} a_{\text{Finishing},j} x_j \leq 5538 \\
& \sum_{j=1}^{111} a_{\text{Assembly},j} x_j \leq 5957 \\
& \sum_{j=1}^{111} a_{\text{QA \& Packaging},j} x_j \leq 6988 \\
& x_j \in \mathbb{Z}_+, \quad j=1,\ldots,111
\end{align*}
\]

Where:
- \( p_j \) is the unit price for component \( j \) (from unit_price.csv, in order C1 to C111).
- \( a_{ij} \) is the unit processing time for component \( j \) in workshop \( i \) (from processing_time_unit.csv, in order).
- \( b_i \) is the total available working hours for workshop \( i \) (from total_working_hours.csv).

Decision variables:
- \( x_j \): number of units to produce of component \( j \), nonnegative integer.

Objective:
- Maximize total output value.

Constraints:
- For each workshop, the total processing time used by all produced components cannot exceed the available working hours.

All data is to be used as given, in the original order, with no sorting or index resetting.