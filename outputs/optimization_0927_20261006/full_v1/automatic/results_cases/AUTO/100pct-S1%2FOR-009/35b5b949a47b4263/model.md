Let $x_i$ be the scale of development per day in area $i$, where $i$ indexes the following areas:

\[
\begin{array}{ll}
\text{Queens} & (\text{Value}=469,\, \text{Weight}=954) \\
\text{Brooklyn} & (\text{Value}=290,\, \text{Weight}=650) \\
\text{Manhattan} & (\text{Value}=236,\, \text{Weight}=961) \\
\text{Bronx} & (\text{Value}=235,\, \text{Weight}=950) \\
\text{Staten Island} & (\text{Value}=745,\, \text{Weight}=379) \\
\text{Harlem} & (\text{Value}=684,\, \text{Weight}=776) \\
\text{Upper East Side} & (\text{Value}=444,\, \text{Weight}=381) \\
\text{Lower Manhattan} & (\text{Value}=172,\, \text{Weight}=808) \\
\text{Midtown} & (\text{Value}=1000,\, \text{Weight}=937) \\
\text{Long Island City} & (\text{Value}=336,\, \text{Weight}=608) \\
\text{Williamsburg} & (\text{Value}=546,\, \text{Weight}=912) \\
\text{Bushwick} & (\text{Value}=535,\, \text{Weight}=391) \\
\text{Flatbush} & (\text{Value}=539,\, \text{Weight}=465) \\
\text{Greenpoint} & (\text{Value}=831,\, \text{Weight}=490) \\
\text{Park Slope} & (\text{Value}=139,\, \text{Weight}=918) \\
\text{Astoria} & (\text{Value}=432,\, \text{Weight}=787) \\
\text{Jackson Heights} & (\text{Value}=627,\, \text{Weight}=347) \\
\text{Flushing} & (\text{Value}=629,\, \text{Weight}=274) \\
\text{Sunnyside} & (\text{Value}=292,\, \text{Weight}=642) \\
\text{Ditmars} & (\text{Value}=978,\, \text{Weight}=130) \\
\end{array}
\]

The overall development capacity is $586$.

The mathematical model is:

\[
\textbf{Objective:} \quad \max \left(
469x_{\text{Queens}} + 290x_{\text{Brooklyn}} + 236x_{\text{Manhattan}} + 235x_{\text{Bronx}} + 745x_{\text{Staten Island}} + 684x_{\text{Harlem}} + 444x_{\text{Upper East Side}} + 172x_{\text{Lower Manhattan}} + 1000x_{\text{Midtown}} + 336x_{\text{Long Island City}} + 546x_{\text{Williamsburg}} + 535x_{\text{Bushwick}} + 539x_{\text{Flatbush}} + 831x_{\text{Greenpoint}} + 139x_{\text{Park Slope}} + 432x_{\text{Astoria}} + 627x_{\text{Jackson Heights}} + 629x_{\text{Flushing}} + 292x_{\text{Sunnyside}} + 978x_{\text{Ditmars}}
\right)
\]

\[
\textbf{Subject to:}
\]

\[
954x_{\text{Queens}} + 650x_{\text{Brooklyn}} + 961x_{\text{Manhattan}} + 950x_{\text{Bronx}} + 379x_{\text{Staten Island}} + 776x_{\text{Harlem}} + 381x_{\text{Upper East Side}} + 808x_{\text{Lower Manhattan}} + 937x_{\text{Midtown}} + 608x_{\text{Long Island City}} + 912x_{\text{Williamsburg}} + 391x_{\text{Bushwick}} + 465x_{\text{Flatbush}} + 490x_{\text{Greenpoint}} + 918x_{\text{Park Slope}} + 787x_{\text{Astoria}} + 347x_{\text{Jackson Heights}} + 274x_{\text{Flushing}} + 642x_{\text{Sunnyside}} + 130x_{\text{Ditmars}} \leq 586
\]

\[
x_i \geq 0, \quad \forall i \in \{\text{all areas listed above}\}
\]

where each $x_i$ is a nonnegative continuous variable representing the scale of development per day in area $i$.