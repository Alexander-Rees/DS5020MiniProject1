# DS5020 Mini Project 1 Individual Reflection

**Name:** Jiaxiao Chen\
**Role:** ML Engineer\
**Date:** 2026-10-12\
**Status:** Draft for review

As ML Engineer, I completed the project that Alex Rees started. Alex's DE1 to DE5 work and exploratory ML1 and ML2 prototype gave me the training matrix and the predeclared k = 100. My completion work added the DE6 rebuild script, a sparse SVD sweep from k = 10 to 200 that kept 100, cosine recommendations for products and customers, an evaluation on products customers first bought after the cutoff, a PCA comparison, a two-component customer map, 30 tests, a reproducible notebook, the docs, memo and slides. I used Claude to assist with implementation and drafting, and Codex for planning and verification.

The results show a real but modest gain. SVD placed a later purchase in the top 10 for 23.2% of 1,913 test customers, against 17.4% for best sellers, and PCA at 22.9% was not reliably different from SVD. Low reconstruction error did not guarantee useful suggestions: raw counts fit the matrix most closely yet hit only 12.9%. The 1/0 version scored 24.9%, but log values stayed because they were fixed before the test. Size depends on the comparison: the 7.3 MB of factors are far below the 161 MB dense matrix but above the 5.0 MB sparse arrays.

Next time, I would choose settings on a training-only validation window before the final test, so a question like 1/0 versus log could be settled without touching the test. I would also compare seasonal windows and add stock availability. Finally, I would record success targets and roles at the start rather than at completion.
