import os
import numpy as np
from pyspark.sql import SparkSession, Row
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.linalg import Vectors
from PIL import Image

# Initialize Spark session
spark = SparkSession.builder \
    .appName("ImageClassification") \
    .config("spark.executor.memory", "8g") \
    .config("spark.driver.memory", "8g") \
    .getOrCreate()

# Set the path to your image folders
train_dir = "/Users/anjalijha/Desktop/UMBC/Semester_2/DS602/DS603/data/Train"

def load_image_data_with_labels(base_path):
    image_data = []

    for label in ['cats', 'dogs', 'horse']:
        folder_path = os.path.join(base_path, label)
        count = 0

        for image_file in os.listdir(folder_path):
            if count >= 250:
                break
            try:
                image_path = os.path.join(folder_path, image_file)
                image = Image.open(image_path).resize((100, 100))
                image_array = np.array(image).astype('float32').flatten()
                vector = Vectors.dense(image_array.tolist())
                image_row = Row(label=label, features=vector)
                image_data.append(image_row)
                count += 1
            except Exception as e:
                print(f"Failed to process {image_path}: {e}")
    
    # ✅ Combine into Spark DataFrame or return None
    if image_data:
        return spark.createDataFrame(image_data)
    else:
        return None

train_data = load_image_data_with_labels(train_dir)
train_data.cache()  # Cache the training data


if train_data is None:
    print("No image data loaded.")
else:
    train_data.show(5)
    assembler = VectorAssembler(inputCols=["features"], outputCol="features_vector")
    train_data = assembler.transform(train_data).select("label", "features_vector")
    train_data.show(5)

    # Convert string labels to numeric labels using StringIndexer
    indexer = StringIndexer(inputCol="label", outputCol="label_index")
    train_data = indexer.fit(train_data).transform(train_data)

    # Split the data into training and validation sets
    train_set, val_set = train_data.randomSplit([0.8, 0.2], seed=42)

    # Initialize Logistic Regression model
    lr = LogisticRegression(featuresCol="features_vector", labelCol="label_index", maxIter=10)
    model = lr.fit(train_set)

    # Make predictions
    predictions = model.transform(val_set)

    # Show sample predictions
    predictions.select("label", "prediction", "probability").show(10)

    # Evaluate model
    evaluator = MulticlassClassificationEvaluator(labelCol="label_index", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    print(f"Validation Accuracy: {accuracy:.2%}")

from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import matplotlib.pyplot as plt

# Evaluate model
evaluator = MulticlassClassificationEvaluator(
    labelCol="label_index",
    predictionCol="prediction",
    metricName="accuracy"
)
accuracy = evaluator.evaluate(predictions)
print(f"Validation Accuracy: {accuracy:.2%}")

# Count correct and incorrect predictions
correct_preds = predictions.filter(predictions.label_index == predictions.prediction).count()
total_preds = predictions.count()
incorrect_preds = total_preds - correct_preds

print(f"Correctly Identified Images: {correct_preds}")
print(f"Incorrectly Identified Images: {incorrect_preds}")

# Plotting results
labels = ['Correct', 'Incorrect']
values = [correct_preds, incorrect_preds]
colors = ['#4CAF50', '#F44336']

plt.figure(figsize=(6, 4))
plt.bar(labels, values, color=colors)
plt.title('Model Prediction Accuracy')
plt.ylabel('Number of Images')
plt.show()
# Save the trained model
model_path = "/Users/anjalijha/Desktop/UMBC/Semester_2/DS602/DS603/saved_model"
model.save(model_path)


from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Get true labels and predicted labels
true_labels = predictions.select('label_index').collect()
predicted_labels = predictions.select('prediction').collect()

# Convert to flat lists
true_labels = [row[0] for row in true_labels]
predicted_labels = [row[0] for row in predicted_labels]

# Compute confusion matrix
cm = confusion_matrix(true_labels, predicted_labels)

# Plot confusion matrix
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Cat", "Dog", "Horse"], yticklabels=["Cat", "Dog", "Horse"])
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
