import os
import numpy as np
from PIL import Image
from pyspark.sql import SparkSession, Row
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegressionModel
from pyspark.ml.linalg import Vectors
import matplotlib.pyplot as plt

# Initialize Spark session
spark = SparkSession.builder.appName("ImageClassification").getOrCreate()

# Set paths for the model and test data
model_path = "/Users/anjalijha/Desktop/UMBC/Semester_2/DS602/DS603/saved_model"
test_path = "/Users/anjalijha/Desktop/UMBC/Semester_2/DS602/DS603/data/Test"

# Load the trained model
loaded_model = LogisticRegressionModel.load(model_path)

# Function to load image data from the test folder
def load_test_images(test_path):
    test_rows = []
    for fname in os.listdir(test_path):
        if fname.endswith(('.jpg', '.png', '.jpeg')):
            fpath = os.path.join(test_path, fname)
            try:
                img = Image.open(fpath).resize((100, 100)).convert("RGB")
                arr = np.array(img).astype('float32').flatten().tolist()
                vector = Vectors.dense(arr)  # Convert the array to a vector
                test_rows.append(Row(filename=fname, features=vector))  # Store as vector in the DataFrame
            except Exception as e:
                print(f"Could not load image: {fpath}, Error: {e}")
    return spark.createDataFrame(test_rows)

# Load the test images
test_df = load_test_images(test_path)

# Assemble the features into a single vector column
assembler = VectorAssembler(inputCols=["features"], outputCol="features_vector")
test_df = assembler.transform(test_df).select("features_vector", "filename")

# Make predictions
predictions = loaded_model.transform(test_df)

# Use Spark's F.when() and F.otherwise() to map predictions to label names
from pyspark.sql import functions as F

predictions_with_labels = predictions.withColumn(
    "label_name", 
    F.when(predictions["prediction"] == 0, F.lit("cats"))
     .when(predictions["prediction"] == 1, F.lit("dogs"))
     .when(predictions["prediction"] == 2, F.lit("horses"))
     .otherwise(F.lit("unknown"))
)

# Show predictions to verify
predictions_with_labels.select("filename", "prediction", "label_name").show(truncate=False)

# Visualize the results
# Get the test images along with their filenames and predictions
image_predictions = predictions_with_labels.select("filename", "label_name", "prediction").collect()

# Define a function to load images
def load_image(image_path):
    try:
        img = Image.open(image_path).resize((100, 100))
        return np.array(img)
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

# Setup the plot size (for example, a grid of 5 rows and 5 columns)
rows = 5
cols = 5

# Create a figure for the plot
fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
axes = axes.flatten()  # Flatten to make indexing easier

# Loop through the image predictions and display them
for i, (image_row, ax) in enumerate(zip(image_predictions, axes)):
    if i >= rows * cols:
        break
    filename = image_row['filename']
    label_name = image_row['label_name']
    
    # Load the image (you'll need the path to the test images)
    image_path = os.path.join(test_path, filename)
    img_array = load_image(image_path)
    
    if img_array is not None:
        ax.imshow(img_array)
        ax.set_title(f"Pred: {label_name}")
        ax.axis('off')

plt.tight_layout()
plt.show()
