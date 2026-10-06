\ Model Original_RAG_FLP
\ LP format - for model browsing. Use MPS format to capture full model detail.
\ Signature: 0x34ca9022fb4a1db5
Minimize
  937.3406504757702 x[S1,C1] + 86.93019411027852 x[S1,C2]
   + 49.3801614788785 x[S2,C1] + 1726.062101362784 x[S2,C2]
   + 105.1815083048055 y[S1] + 112.1842388512642 y[S2]
Subject To
 demand[C1]: x[S1,C1] + x[S2,C1] = 7564
 demand[C2]: x[S1,C2] + x[S2,C2] = 5246
 activation[S1]: x[S1,C1] + x[S1,C2] - 12810 y[S1] <= 0
 activation[S2]: x[S2,C1] + x[S2,C2] - 12810 y[S2] <= 0
Bounds
Binaries
 y[S1] y[S2]
End
