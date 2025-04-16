from flask import Flask, request, jsonify
import pandas as pd
from flask_cors import CORS
from model.recommendation import recommend_food, recommend_based_on_country_and_profit, collaborative_recommendation_with_similarity, evaluate_model

app = Flask(__name__)
CORS(app)

# def load_dataset(file_path):
#     data = pd.read_csv(file_path)
#     print(f"Dataset loaded with shape: {data.shape}")
#     return data

# data_1_path = 'data/Updated_FoodRecommandation_With_Prices (1).csv'
# data_2_path = 'data/Large_Generated_Orders_Dataset (1) (1).csv'

# data_1 = load_dataset(data_1_path)
# data_2 = load_dataset(data_2_path)
# def preprocess_data(data):
#     data['Preparation Time (Minutes)'] = pd.to_numeric(data['Preparation Time (Minutes)'], errors='coerce')
#     data['Ingredient Count'] = pd.to_numeric(data['Ingredient Count'], errors='coerce')
#     data['Selling Price (EGP)'] = pd.to_numeric(data['Selling Price (EGP)'], errors='coerce')
#     data['Cost Price (EGP)'] = pd.to_numeric(data['Cost Price (EGP)'], errors='coerce')
#     return data.dropna()

# data_1 = preprocess_data(data_1)


@app.route('/recommend_food', methods=['POST'])
def recommend_food_endpoint():
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "Missing request body"}), 400

        prep_time = data.get('prep_time')
        num_ingredients = data.get('num_ingredients')
        country = data.get('country')
        selling_price = data.get('selling_price')
        cost_price = data.get('cost_price')
        n_recommendations = data.get('n_recommendations', 5)

        if not all([prep_time, num_ingredients, country, selling_price, cost_price]):
            return jsonify({"error": "Missing required parameters"}), 400

        try:
            prep_time = float(prep_time)
            num_ingredients = int(num_ingredients)
            selling_price = float(selling_price)
            cost_price = float(cost_price)
            n_recommendations = int(n_recommendations)
        except ValueError:
            return jsonify({"error": "Invalid data type in parameters"}), 400

        if prep_time < 0 or num_ingredients < 0 or selling_price < 0 or cost_price < 0:
            return jsonify({"error": "Values cannot be negative"}), 400
        if n_recommendations <= 0:
            return jsonify({"error": "n_recommendations must be greater than zero"}), 400

        recommendations = recommend_food(prep_time, num_ingredients, country, selling_price, cost_price, n_recommendations)

        return jsonify(recommendations), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500



@app.route('/recommend_profit', methods=['POST'])
def recommend_profit_endpoint():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Missing request body"}), 400

        selected_country = data.get('country')
        top_n = data.get('top_n', 5)

        if not selected_country:
            return jsonify({"error": "Missing required parameter: country"}), 400

        try:
            top_n = int(top_n)
        except ValueError:
            return jsonify({"error": "Invalid data type for top_n, must be an integer"}), 400

        if top_n <= 0:
            return jsonify({"error": "top_n must be greater than zero"}), 400

        recommendations = recommend_based_on_country_and_profit(selected_country, top_n)

        return jsonify(recommendations), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/collaborative_recommendation', methods=['POST'])
def collaborative_recommendation_endpoint():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Missing request body"}), 400

        country = data.get('country')
        n_recommendations = data.get('n_recommendations', 5)

        if not country:
            return jsonify({"error": "Missing required parameter: country"}), 400

        try:
            n_recommendations = int(n_recommendations)
        except ValueError:
            return jsonify({"error": "Invalid data type for n_recommendations, must be an integer"}), 400

        if n_recommendations <= 0:
            return jsonify({"error": "n_recommendations must be greater than zero"}), 400

        recommendations = collaborative_recommendation_with_similarity(country, n_recommendations)

        if not recommendations:
            return jsonify({"message": "No recommendations found for the given country"}), 404

        return jsonify(recommendations), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/evaluate_model', methods=['GET'])
def evaluate_model_endpoint():
    avg_mean_distance, accuracy = evaluate_model()
    return jsonify({
        'average_mean_distance': avg_mean_distance,
        'accuracy': accuracy
    })

if __name__ == '__main__':
    app.run(debug=True)
