\ Model Original_RAG_TP
\ LP format - for model browsing. Use MPS format to capture full model detail.
\ Signature: 0xbf2376827e5f1f2a
Minimize
  0.63 x[S1,C1] + 49.71 x[S1,C2] + 33.75 x[S1,C3] + 1570.67 x[S1,C4]
   + 605.47 x[S2,C1] + 64.53 x[S2,C2] + 478.47 x[S2,C3] + 887.04 x[S2,C4]
   + 1139.04 x[S3,C1] + 4.78 x[S3,C2] + 1805.62 x[S3,C3] + 1302.89 x[S3,C4]
Subject To
 demand[C1]: x[S1,C1] + x[S2,C1] + x[S3,C1] >= 11
 demand[C2]: x[S1,C2] + x[S2,C2] + x[S3,C2] >= 1148
 demand[C3]: x[S1,C3] + x[S2,C3] + x[S3,C3] >= 54
 demand[C4]: x[S1,C4] + x[S2,C4] + x[S3,C4] >= 833
 supply[S1]: x[S1,C1] + x[S1,C2] + x[S1,C3] + x[S1,C4] <= 4
 supply[S2]: x[S2,C1] + x[S2,C2] + x[S2,C3] + x[S2,C4] <= 575
 supply[S3]: x[S3,C1] + x[S3,C2] + x[S3,C3] + x[S3,C4] <= 1504
Bounds
End
