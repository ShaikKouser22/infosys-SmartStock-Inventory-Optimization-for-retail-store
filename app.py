from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import os

try:
    import  openai
except Exception:
    openai = None


def predict_product_status(Stock_Quantity, Reorder_Level):
    """Predict product status based on inventory levels."""
    if Stock_Quantity <= 10 and Reorder_Level >= 50:
        return "Backordered"
    elif Stock_Quantity >= 90 and Reorder_Level <= 30:
        return "Discontinued"
    else:
        return "Active"


# 1. Initialize Flask app
app = Flask(__name__)

# 2. Load model and encoders once when the app starts (use absolute paths)
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, "grocery_model (2).pkl")  
encoders_path = os.path.join(base_dir, "label_encoders (2).pkl")  


model = None
encoders = None

try:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    model = joblib.load(model_path)
    print(f"Model loaded successfully from {model_path}")

    if os.path.exists(encoders_path):
        encoders = joblib.load(encoders_path)
        print(f"Encoders loaded successfully from {encoders_path}")
    else:
        encoders = None
        print("Label encoders file not found, proceeding without encoders.")

except Exception as e:
    print(f"Error loading model files: {e}")

# LLM Configuration
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
 
 #llm configuration

def get_llm_prompt(status):
    """
    Return a short actionable message for the given status.
    Uses OpenAI if OPENAI_API_KEY is set and openai package is available,
    otherwise returns a canned fallback message.
    """
    fallbacks = {
        "Backordered": "Product is backordered — update customers about expected delivery delays.",
        "Discontinued": "Product appears discontinued — mark item as inactive in catalog, halt all reorders, and inform sales/marketing team.",
        "Active": "Product is active — maintain regular inventory monitoring and reorder based on demand patterns and sales volume."
    }

    # If openai library not installed or API key not set, use fallback
    if not openai or not OPENAI_KEY:
        print("OpenAI not configured, using fallback message.")
        return fallbacks.get(status, "No additional guidance available.")
    
    # Call OpenAI API to get a short actionable message
    try:
        openai.api_key = OPENAI_KEY
        prompt = (
            f"You are an inventory management assistant. Provide one concise, actionable sentence (max 25 words) "
            f"for the inventory status: {status}. Make it practical, direct, and useful for warehouse managers."
        )
        
        resp = openai.ChatCompletion.create(
            model=LLM_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert inventory management consultant. Provide short, actionable guidance."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=60,
            temperature=0.6
        )
        
        text = resp.choices[0].message['content'].strip()
        print(f"OpenAI response: {text}")
        return text if text else fallbacks.get(status)
    
    except Exception as e:
        print(f"OpenAI API error: {e}")
        return fallbacks.get(status)
# 3. Define routes

@app.route('/', methods=['GET'])
def home():
    """Renders the HTML form for prediction."""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Handles POST requests to make a prediction."""
    try:
        # Accept JSON or form data
        input_data = request.get_json(silent=True)
        if not input_data:
            input_data = request.form.to_dict()

        if not input_data:
            return jsonify({"error": "No input data provided."}), 400

        # Ensure required fields present
        if 'Stock_Quantity' not in input_data or 'Reorder_Level' not in input_data:
            return jsonify({
                "error": "Missing required fields: Stock_Quantity and Reorder_Level."
            }), 400

        # Create DataFrame
        df = pd.DataFrame([input_data])

        # Apply label encoders if available
        if encoders:
            for col, le in encoders.items():
                if col in df.columns:
                    try:
                        df[col] = le.transform(df[col].astype(str))
                    except Exception as e:
                        print(f"Encoding error for column {col}: {e}")

        # Extract numeric values
        try:
            sq = int(df['Stock_Quantity'].iloc[0])
            rl = int(df['Reorder_Level'].iloc[0])
        except (ValueError, TypeError) as e:
            return jsonify({
                "error": f"Stock_Quantity and Reorder_Level must be valid integers. Error: {str(e)}"
            }), 400

        # Get prediction
        prediction = predict_product_status(sq, rl)

        # Get LLM-generated prompt/message (or fallback)
        try:
            message = get_llm_prompt(prediction)
            if not message or not isinstance(message, str):
                message = "No guidance available."
        except Exception as e:
            print(f"LLM error: {e}")
            message = "No guidance available."


        # Return prediction and LLM message
        return jsonify({
            "prediction": prediction,
            "message": message
        })

    except Exception as e:
        return jsonify({
            "error": f"Prediction failed. Details: {str(e)}"
        }), 400


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None,
        "encoders_loaded": encoders is not None,
        "openai_configured": openai is not None and OPENAI_KEY is not None
    })


if __name__ == '__main__':
    app.run(debug=True, port=5000)
