package com.smartscity.spark;

/**
 * 2172
 * Your submission scored 0.79425, which is not an improvement of your best score. Keep trying!
 * Score  : 0.8978765219058574
 *          by  pom spark-core_2.10  version:2.0.0
 */

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.RandomForestClassifier;
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator;
import org.apache.spark.ml.feature.*;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.tuning.CrossValidator;
import org.apache.spark.ml.tuning.CrossValidatorModel;
import org.apache.spark.ml.tuning.ParamGridBuilder;
import org.apache.spark.sql.*;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.api.java.UDF1;
import org.apache.spark.sql.api.java.UDF4;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

import java.util.regex.Matcher;
import java.util.regex.Pattern;

import static org.apache.spark.sql.functions.*;

public class RandomForestsKaggle {

    public static void main(String[] args) {
        Logger.getLogger( "org" ).setLevel( Level.ERROR );
        Logger.getLogger( "akka" ).setLevel( Level.ERROR );

        SparkConf sparkConf = new SparkConf()
                .setMaster( "local[*]" )
                .setAppName( "Titanic Spark" );
        JavaSparkContext javaSparkContext = new JavaSparkContext( sparkConf );
        SQLContext sqlContext = new SQLContext( javaSparkContext );

		/*--------------------------------------------------------------------------
		Loading  train Data
		--------------------------------------------------------------------------*/
        //Create the schema for the data to be loaded into Dataset.

        StructType dataSchema = DataTypes
                .createStructType(new StructField[] {
                        DataTypes.createStructField("PassengerId", DataTypes.IntegerType, false),
                        DataTypes.createStructField("Survived", DataTypes.IntegerType, false),
                        DataTypes.createStructField("Pclass", DataTypes.IntegerType, false),
                        DataTypes.createStructField("Name", DataTypes.StringType, false),
                        DataTypes.createStructField("Sex", DataTypes.StringType, false),
                        DataTypes.createStructField("Age", DataTypes.DoubleType, false),
                        DataTypes.createStructField("SibSp", DataTypes.IntegerType, false),
                        DataTypes.createStructField("Parch", DataTypes.IntegerType, false),
                        DataTypes.createStructField("Ticket", DataTypes.StringType, false),
                        DataTypes.createStructField("Fare", DataTypes.DoubleType, false),
                        DataTypes.createStructField("Bin", DataTypes.StringType, false),
                        DataTypes.createStructField("Embarked", DataTypes.StringType, false)
                });


        Dataset<Row> trainDf = sqlContext.read().option( "header", "true" ).option( "inferSchema", "true" ).csv( "/Users/apple/git/kaggle/train.csv" );
        trainDf.show( 5 );
        trainDf.printSchema();
		/*--------------------------------------------------------------------------
		Cleanse Data
		--------------------------------------------------------------------------*/


        System.out.println( "Number of passengers in training data: " + trainDf.count() );

        Dataset<Row> protrainDf = processData( trainDf ,sqlContext);
        protrainDf.show();


		/*--------------------------------------------------------------------------
		Analyze Data
		--------------------------------------------------------------------------*/

        //Perform correlation analysis
        for ( StructField field : dataSchema.fields() ) {
            if ( ! field.dataType().equals(DataTypes.StringType)) {
                System.out.println( "Correlation between Survived and " + field.name()
                        + " = " + protrainDf.stat().corr("Survived", field.name()) );
            }
        }

        // Converting string labels into indices.
        StringIndexer embarkedIndexer = new StringIndexer().setInputCol( "Embarked" ).setOutputCol( "EmbarkedIndexed" ).setHandleInvalid( "skip" );
        StringIndexer sexIndexer = new StringIndexer().setInputCol( "Sex" ).setOutputCol( "SexIndexed" ).setHandleInvalid( "skip" );
        StringIndexer survivedIndexer = new StringIndexer().setInputCol( "Survived" ).setOutputCol( "SurvivedLabel" ).setHandleInvalid( "skip" );
        StringIndexer TitleIndexer = new StringIndexer().setInputCol( "title" ).setOutputCol( "TitleIndexed" ).setHandleInvalid( "skip" );


        double[] fareSplits = new double[]{0.0, 10.0, 20.0, 30.0, 40.0, Double.POSITIVE_INFINITY};
        Bucketizer fareBucketize = new Bucketizer()
                .setInputCol("Fare")
                .setOutputCol("FareBucketed")
                .setSplits( fareSplits );


        // Creating dummy columns
        OneHotEncoder embEncoder = new OneHotEncoder().setInputCol( "EmbarkedIndexed" ).setOutputCol( "EmbarkedVec" );
        OneHotEncoder sexEncoder = new OneHotEncoder().setInputCol( "SexIndexed" ).setOutputCol( "SexVec" );
        OneHotEncoder titleEncoder = new OneHotEncoder().setInputCol( "TitleIndexed" ).setOutputCol( "TitleVec" );
        // The vector assembler creates a feature column where it combines all the required features at one place.
        VectorAssembler assembler = new VectorAssembler()
                .setInputCols( new String[]{"Pclass", "SexVec", "AgeCat", "SibSp", "Parch", "FareBucketed", "EmbarkedVec", "Family", "Child", "Mom","TitleIndexed"} )
                .setOutputCol( "features" );

        // Performing PCA

        PCA pca = new PCA().setK( 10 ).setInputCol( "features" ).setOutputCol( "pcaFeatures" );

        // Train a DecisionTree model.

        RandomForestClassifier rfc = new RandomForestClassifier()
                .setFeaturesCol( "pcaFeatures" )
                .setLabelCol( "SurvivedLabel" );

                Pipeline pipeline = new Pipeline()
                .setStages( new PipelineStage[]{embarkedIndexer, sexIndexer, survivedIndexer, TitleIndexer , fareBucketize , embEncoder, sexEncoder, assembler, pca, rfc} );
        // Specifying grid search parameters

//        ParamMap[] paramGrid = new ParamGridBuilder()
//                .addGrid( rfc.maxBins(), new int[]{25, 30, 50} )
//                .addGrid( rfc.maxDepth(), new int[]{4, 6, 10} )
//                .build();

//        ParamMap[] paramGrid = new ParamGridBuilder()
//                .addGrid( rfc.maxBins(), new int[]{10, 100, 1000} )
//                .addGrid( rfc.maxDepth(), new int[]{4, 6, 10} )
//                .build();

//        ParamMap[] paramGrid = new ParamGridBuilder()
//                .addGrid( rfc.numTrees(), new int[]{3, 5, 9} )
//                .addGrid( rfc.maxBins(), new int[]{10, 100, 1000} )
//                .addGrid( rfc.maxDepth(), new int[]{4, 6, 10} )
//                .build();  降低

        ParamMap[] paramGrid = new ParamGridBuilder()
                .addGrid( rfc.numTrees(), new int[]{10, 20, 30} )
                .addGrid( rfc.maxBins(), new int[]{10, 100, 1000} )
                .addGrid( rfc.maxDepth(), new int[]{4, 6, 10} )
                .build();

//        ParamMap[] paramGrid = new ParamGridBuilder()
//                .addGrid( rfc.numTrees(), new int[]{3, 5, 9} )
//                .addGrid( rfc.maxBins(), new int[]{10, 100, 1000} )
//                .addGrid( rfc.maxDepth(), new int[]{3, 5, 7, 10} )
//                .build();       // Score  : 0.9086279867138112

        BinaryClassificationEvaluator evaluator = new BinaryClassificationEvaluator()
                .setLabelCol( "SurvivedLabel" );
        // Cross Validation using K-folds approach
        CrossValidator cv = new CrossValidator()
                .setEstimator( pipeline )
                .setEvaluator( evaluator )
                .setEstimatorParamMaps( paramGrid )
                .setNumFolds( 10);

        CrossValidatorModel cvModel = cv.fit( protrainDf );
        Dataset<Row> bestModel = cvModel.bestModel().transform( protrainDf );
        bestModel.select( "PassengerId", "prediction" ).show( 30 );

        Double score = evaluator.evaluate(bestModel);
        // Score  : 0.8939885384377763
        // Score  : 0.8942361976586886
        // Score  : 0.8900206648984331
        // Score  : 0.8978765219058574
        System.out.println( "Score  : " + score);
        System.out.println( "Before testing......." );


        // processing the test data...
        Dataset<Row> testDf = sqlContext.read().option( "header", "true" ).option( "inferSchema", "true" ).csv( "/Users/apple/git/kaggle/test.csv" );
        testDf.show( 5 );
        testDf.printSchema();

        Dataset<Row> InptestDf = processData( testDf,sqlContext );
        // Prediction on test data.
        Dataset<Row> predTest = cvModel.transform( InptestDf );
        predTest = predTest.withColumn( "prediction", predTest.col( "prediction" ).cast( "int" ) );
        predTest = predTest.withColumnRenamed( "prediction", "Survived" );
        predTest.select( "PassengerId", "Survived" ).show( 30 );
        // Saving to a File
        predTest.select("PassengerId", "Survived").write().mode("overwrite").option("header", true).csv("spark_2.0.0");

    }

    //Feature Engineering....

    public static Dataset<Row> processData (Dataset < Row > data, SQLContext sqlContext)
    {
        data = data.drop( "Ticket" ).drop( "Cabin" );
        Double ageMean = data.select( mean( "Age" ) ).head().getDouble( 0 );
        Double fareMean = data.select( mean( "Fare" ) ).head().getDouble( 0 );
        data = data.na().fill( ageMean, new String[]{"Age"} );
        data = data.na().fill( fareMean, new String[]{"Fare"} );


        data = data.withColumn( "Family", data.col( "Parch" ).$plus( data.col( "SibSp" )).$plus( 1 ));


        sqlContext.udf().register( "childInd", new UDF1<Double, Integer>() {
            @Override
            public Integer call(Double val) {
                if (val < 16)
                    return 1;
                else
                    return 0;
            }
        }, DataTypes.IntegerType );

     sqlContext.udf().register( "FindTitle", new UDF1<String, String>() {
            @Override
            public String call(String Name) throws Exception {
                String title= Name;

                Pattern p = Pattern.compile( "(Dr|Mrs?|Ms|Miss|Master|Rev|Capt|Mlle|Col|Major|Sir|Lady|Mme|Don|Mr.)" );
                Matcher m = p.matcher( Name);

                if (m.find()) {
                    title = m.group(1);
                }


                switch (title) {
                    case "Mrs?":
                    case "Dona":
                    case "Mme":
                    case "Lady":
                        title = "Mrs";
                        break;
                    case "Rev":
                    case "Col":
                    case "Major":
                    case "Capt":
                    case "Master":
                    case "Jonkheer":
                    case "Sir":
                    case "Don":
                    case "Dr" :
                        title = "Mr";
                        break;
                    case "Mlle":
                    case "Countess":
                    case "Ms":
                        title = "Miss";
                        break;
                }
                return title;
            }

        }, DataTypes.StringType );



        sqlContext.udf().register( "AgeCateg", new UDF1<Double, Integer>() {
            @Override
            public Integer call(Double age) {
                if (age > 0 && age < 20) {
                    return 1;
                } else if (age >= 20 && age < 30) {
                    return 2;
                } else if (age >= 30 && age < 50) {
                    return 3;
                } else if (age >= 50) {
                    return 4;
                }
                return 0;
            }
        }, DataTypes.IntegerType );

        sqlContext.udf().register( "momInd", new UDF4<Double, String, Integer, String, Integer>() {
            @Override
            public Integer call(Double age, String gender, Integer parch, String name) {
                if ((age > 17) && (gender.equals( "female" )) && (parch > 0) && (!name.contains( "Miss" )))
                    return 1;
                else
                    return 0;
            }
        }, DataTypes.IntegerType );


        data = data.withColumn( "Child", callUDF( "childInd", data.col( "Age" ) ) );
        data = data.withColumn( "Mom", callUDF( "momInd", data.col( "Age" ), data.col( "Sex" ), data.col( "Parch" ), data.col( "Name" ) ) );
      data = data.withColumn( "title", callUDF( "FindTitle", data.col( "Name" ) ) );
        data = data.withColumn( "AgeCat", callUDF( "AgeCateg", data.col( "Age" ) ) );


        return data;





    }




}


