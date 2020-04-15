===============================
Spark Tree Interpreter (Scala & MLlib port of https://github.com/andosa/treeinterpreter)
===============================

This is a Scala & MLlib port of https://github.com/andosa/treeinterpreter.

Free software: BSD license

Dependencies
------------
Spark 2.3


Usage and Tests
-----
Given a trained RandomForestModel/DecisionTreeModel and an Dataset[Row], we have

```
Interp.interpretModelRf(model, trainingData)
Interp.interpretModelDt(model, trainingData)
```
yields
``` 
 prediction:20.897107560022445
 bias:22.05948146469634
 contributionMap Map(Some(4) -> 0.38498427854727063, Some(6) -> 0.6947299575790253, Some(12) -> -0.7793909217869917, Some(5) -> -0.9248027735005142, Some(2) -> -1.01193915560854)
 sumOfTerms: 20.42306284992659
```

The sum of bias and feature contributions should equal the prediction, but due to floating point arithmetic they will be slightly off.

To run tests, just run `sbt test` .

