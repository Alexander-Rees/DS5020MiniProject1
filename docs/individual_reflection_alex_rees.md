# DS5020 Mini Project 1 Individual Reflection

**Name:** Alex Rees
**Role:** Data Engineer
**Date:** 2026-10-13

I was the Data Engineer, so I loaded both yearly sheets, and wrote the data profile. Then I cleaned and ran these in a fixed order: cancellations, non positive quantity, non positive price, non product codes such as postage and vouchers, and lines with no customer ID. That left 802,634 rows. I split the history on 9 October 2011, kept 684,071 earlier rows for training, and held out 72,919 later purchases of products that customer had not bought before. Repeat buys in that window stayed out of both files. The training matrix is 5,473 customers by 3,687 products, with log(1 + total quantity) in each filled cell, and a product stays only if at least 10 training customers bought it. It is 97.95% sparse: 5.0 MB stored that way, against 161.4 MB if every zero were kept. I also ran a full SVD and chose k = 100, because the error was still falling at 200 but each added component helped less.

I learned about the difference between SVD and PCA. SVD left the zeros as zeros, so they still meant “did not buy.” PCA subtracts the average first, so those zeros become small negatives, “bought less than average.” With 98% of the cells empty, that changes the data. On the test set the two still scored almost the same, 23.2% and 22.9%.

Next time I would drop the exact duplicate lines, and the two huge orders whose cancellations we removed while the original sales stayed in training. I would fit the sparse SVD from the start, and agree on the success targets with my partner on the first day.
