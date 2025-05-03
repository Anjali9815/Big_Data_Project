from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegressionModel
from pyspark.sql.functions import udf
from pyspark.ml.linalg import Vectors, VectorUDT
from PIL import Image
import numpy as np
import os
from pyspark.sql import Row

# Initialize Spark session
spark = SparkSession.builder.appName("ImageClassification_test").getOrCreate()

# Path to test images
test_dir = "/Users/anjalijha/Desktop/UMBC/Semester_2/DS602/DS603/data/Test"

# Load the saved model
model_path = "/Users/anjalijha/Desktop/UMBC/Semester_2/DS602/DS603/saved_model"
model = LogisticRegressionModel.load(model_path)

# Function to load test images and convert to feature vectors
def load_test_image_data(base_path):
    image_data = []

    for image_file in os.listdir(base_path):
        # Extract label from the filename (assuming format like 'cat.4242.jpg')
        try:
            label = image_file.split('.')[0]  # Get the part before the dot (e.g., 'cat', 'dog', etc.)
            if label not in ['cat', 'dog', 'horse']:  # Handle unknown labels
                continue  # Skip images with unrecognized labels
            
            # Load image and process
            image_path = os.path.join(base_path, image_file)
            image = Image.open(image_path).resize((100, 100))
            image_array = np.array(image).astype('float32').flatten()
            vector = Vectors.dense(image_array.tolist())
            image_row = Row(label=label, features=vector)
            image_data.append(image_row)

        except Exception as e:
            print(f"Failed to process {image_file}: {e}")

    # ✅ Return Spark DataFrame
    if image_data:
        return spark.createDataFrame(image_data)
    else:
        return None

# Load the test data
test_data = load_test_image_data(test_dir)

if test_data is None:
    print("No image data loaded.")
else:
    test_data.show(5)
    
    # Convert features to the vector column (same as training)
    assembler = VectorAssembler(inputCols=["features"], outputCol="features_vector")
    test_data = assembler.transform(test_data).select("label", "features_vector")
    
    # Convert string labels to numeric labels using StringIndexer (same as during training)
    indexer = StringIndexer(inputCol="label", outputCol="label_index")
    test_data = indexer.fit(test_data).transform(test_data)
    
    # Prepare test set
    test_data = test_data.select("features_vector", "label_index", "label")

    # Make predictions using the loaded model
    predictions = model.transform(test_data)
    
    # Show sample predictions
    predictions.select("label", "prediction", "probability").show(10)

    from pyspark.sql import functions as F

    # Define label map (mapping numeric predictions to the corresponding class names)
    label_map = {0: 'cat', 1: 'dog', 2: 'horse'}

    # Add a new column for the mapped labels
    predictions_with_labels = predictions.withColumn(
        "label_name",
        F.when(F.col("prediction") == 0, "cat")
        .when(F.col("prediction") == 1, "dog")
        .when(F.col("prediction") == 2, "horse")
        .otherwise("unknown")  # For any unrecognized prediction, mark as "unknown"
    )

    # Show the predictions with labels
    predictions_with_labels.select("label", "label_name", "prediction", "probability").show(10)
