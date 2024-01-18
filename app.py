from flask import Flask, render_template, request
import csv
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from werkzeug.utils import secure_filename
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
app = Flask(__name__)

# Load the saved model
model = tf.keras.models.load_model('food_augment.h5')

# Load the class labels from the generator used during training
class_labels = list(np.load('class_aug.npy', allow_pickle=True).item().keys())

# Function to make predictions on a new image
def predict_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Create a batch

    # Preprocess the input image
    img_array /= 255.0

    # Make predictions
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])

    # Return the predicted class and its label
    return class_labels[predicted_class]

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the image file
        image_file = request.files['image']
        if image_file:
            # Save the image in the 'static' folder
            image_path = secure_filename(image_file.filename)
            image_file.save(image_path)

            # Predict food from the image
            pred_food = predict_image(image_path)

            # Personal details from the form
            gender = request.form['gender']
            age = int(request.form['age'])
            height = float(request.form['height'])
            weight = float(request.form['weight'])
            activity_level = request.form['activity_level']
            medical_conditions = request.form['medical_conditions']

            # Collect insights and statistics during analysis
            insights = []
            bmr = 0  # Initialize with default value
            total_calories = 0  # Initialize with default value
            carb_ratio = 0  # Initialize with default value

            # BMR Calculation
            if gender == "male":
                bmr = 10 * weight + 6.25 * height - 5 * age + 5
            else:
                bmr = 10 * weight + 6.25 * height - 5 * age - 161

            # Activity factor
            activity_factors = {"sedentary": 1.2, "moderate": 1.5, "active": 1.75, "low": 1.2}
            activity_factor = activity_factors[activity_level]
            total_calories = activity_factor * bmr

            # Nutrition data
            with open('food.csv', 'r', encoding='utf') as file:
                reader = csv.reader(file)
                food_data = list(reader)

            food_item = pred_food
            for item in food_data:
                if item[0].lower() == food_item.lower():
                    # Extract details
                    protein = float(item[4])
                    carbs = float(item[2])
                    fats = float(item[3])
                    gi = float(item[1])

                    # Overall macros
                    total_macros = protein + carbs + fats
                    calories = protein * 4 + carbs * 4 + fats * 9

                    # Macro ratios
                    protein_ratio = protein / total_macros
                    carb_ratio = carbs / total_macros

            # List of conditions
            conditions = ["diabetes", "PCOS"]
            df = pd.read_csv('food.csv')
            columns_of_interest = ['protein', 'fat', 'carbohydrates', 'glycemic index']
            features = df[columns_of_interest]
            if medical_conditions.lower() in conditions:
                if gi > 55 and (medical_conditions.lower() == "diabetes" or medical_conditions.lower() == "pcos"):
                    print("erujiyhguikedrghfqwegbukfghqekughuklqegukeqlkug")# Insights for diabetes
                    with open('food.csv', 'r', encoding='utf-8') as file:
                        reader = csv.DictReader(file)
                        pizza_index = df.index[df['food'] == food_item].tolist()[0]
                        cosine_sim_to_pizza = cosine_similarity([features.iloc[pizza_index]], features)
                        top_5_similar_indices = cosine_sim_to_pizza.argsort()[0][-4:-1][::-1]
                        top_5_similar_food_names = df['food'].iloc[top_5_similar_indices]
                        top_5_similar_glycemic_values = df['glycemic index'].iloc[top_5_similar_indices]
                        for food_name, glycemic_value in zip(top_5_similar_food_names.values, top_5_similar_glycemic_values.values):
                            
                            if glycemic_value < 55:
                                print(f"{food_name} with glycemic index: {glycemic_value}")
                                c=[food_name, glycemic_value]
                                insights.append(f"{c[0]} with glycemix index: {c[1]}")
            else :
                with open('food.csv', 'r', encoding='utf-8') as file:
                    reader = csv.DictReader(file)
                    pizza_index = df.index[df['food'] == food_item].tolist()[0]
                    pizza_glycemic_index = df['glycemic index'].iloc[pizza_index]
                    cosine_sim_to_pizza = cosine_similarity([features.iloc[pizza_index]], features)
                    top_5_similar_indices = cosine_sim_to_pizza.argsort()[0][-4:-1][::-1]
                    top_5_similar_food_names = df['food'].iloc[top_5_similar_indices]
                    top_5_similar_glycemic_values = df['glycemic index'].iloc[top_5_similar_indices]
                    for food_name, glycemic_value in zip(top_5_similar_food_names.values, top_5_similar_glycemic_values.values):
                        if glycemic_value < 100000000000000000000:
                            print(f"{food_name} with glycemic index: {glycemic_value}")
                            c=[food_name, glycemic_value]
                            insights.append(f"{c[0]} with glycemix index: {c[1]}")
                    

            print("Analysis tailored to your conditions!")

            if weight / height ** 2 > 25:
                if carb_ratio > 0.45:
                    insight = "Consider lower carb option for weight management"
                    insights.append(insight)

            # Render the result page with food prediction, insights, and statistics
            return render_template('result.html', food=pred_food, insights=insights, bmr=bmr, total_calories=total_calories,
                                   carb_ratio=carb_ratio, protein=protein, carbs=carbs, fats=fats, gi=gi, protein_ratio=protein_ratio)

    # Render the input page
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
