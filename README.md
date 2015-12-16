===============================
Spark Tree Interpreter (Scala & MLlib port of https://github.com/andosa/treeinterpreter)
===============================

This is a Scala & MLlib port of https://github.com/andosa/treeinterpreter.

Free software: BSD license

Dependencies
------------
Spark 1.4.1+

Installation
------------
TBD

Usage and Tests
-----
Given RandomForestModel or DecisionTreeModel and data set trainingData
`Interp.interpretModel(model, trainingData)`
yields
```
 bias:22.05948146469634
 prediction:20.897107560022445
 contributionMap Map(Some(4) -> 0.38498427854727063, Some(6) -> 0.6947299575790253, Some(12) -> -0.7793909217869917, Some(5) -> -0.9248027735005142, Some(2) -> -1.01193915560854)
 sumOfTerms: 20.42306284992659
```

The sum of bias and feature contributions should equal the prediction, but due to floating point arithmetic they will be slightly off.


