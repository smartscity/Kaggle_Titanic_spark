# SparkML 2.0 vs SparkML 2.4.0


###  Introduce

 This case is Titanic Competitions on the Kaggle.
 https://www.kaggle.com/c/titanic

### Module spark_2.0.0  `spark-core_2.10  version 2.0.0 `
    source ../spark_2.0.0

### Module spark_2.4.0  `spark-core_2.11  version 2.4.0 `
    source ../spark_2.4.0

### Question
 I use different versions of spark to analyze random forest scores..

* spark-core_2.10  `version 2.0.0 `
    *  RandomForestsKaggle Score = 0.8978765219058574
* spark-core_2.11  `version 2.4.0 `
    *  RandomForestsKaggle Score = 0.8886987035251259

### Conclusion
 After upgrading the spark version(`version 2.4.0`), the random forest score dropped(`0.01`).
