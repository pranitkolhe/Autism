from flask import Flask, request, jsonify
import pandas as pd
import joblib
import traceback

app = Flask(__name__)


try:
    with open('model.pkl', 'rb') as model_file:
        model = joblib.load(model_file)
    print("✅ New model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

@app.route('/predict_ml', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model is not loaded'}), 500

    try:

        data_from_form = request.get_json()


        REQUIRED_FEATURES = [
            'Qchat_10_Score',
            'Ethnicity',
            'Sex',
            'A6',
            'A9',
            'A5',
            'A7',
            'A1',
            'A2'
        ]


        input_df = pd.DataFrame([data_from_form])
        

        input_df = input_df[REQUIRED_FEATURES]
        

        prediction = model.predict(input_df)

        report_text = ""
        if prediction[0] == 'Yes' or prediction[0] == 1:
             report_text = """
Analysis: Indicators Consistent with ASD Traits

Conclusion: The model's prediction is POSITIVE for ASD Traits.
(Full report text here)
"""
        else:
             report_text = """
Analysis: Indicators Not Consistent with ASD Traits

Conclusion: The model's prediction is NEGATIVE for ASD Traits.
(Full report text here)
"""
        return jsonify({'prediction': report_text.strip()})

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': f"An error occurred: {e}"}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)






#///////////////////////////////////////////////////////////////////////////////////////////////////
# from flask import Flask, request, jsonify
# import pickle
# import numpy as np
# import pandas as pd
# import traceback

# app = Flask(__name__)

# # Load your 27-feature model
# try:
#     with open('model.pkl', 'rb') as model_file:
#         model = pickle.load(model_file)
#     print("✅ Full 27-feature model loaded successfully!")
# except Exception as e:
#     print(f"❌ Error loading model: {e}")
#     model = None

# @app.route('/predict_ml', methods=['POST'])
# def predict():
#     if model is None:
#         return jsonify({'error': 'Model is not loaded'}), 500

#     try:
#         data = request.get_json()

#         # Define the full list of 27 feature columns your model expects
#         ALL_FEATURE_COLUMNS = [
#             'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10_Autism_Spectrum_Quotient',
#             'Social_Responsiveness_Scale', 'Age_Years', 'Qchat_10_Score', 'Speech Delay/Language Disorder',
#             'Learning disorder', 'Genetic_Disorders', 'Depression', 'Global developmental delay/intellectual disability',
#             'Social/Behavioural Issues',
#             'Childhood Autism Rating Scale', 'Anxiety_disorder',
#             'Sex', 'Ethnicity', 'Jaundice', 'Family_mem_with_ASD', 'Who_completed_the_test',
#             "CASE_NO_PATIENT'S"
#         ]

#         # Define the 9 features we get from the form
#         form_features = [
#             'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7',
#             'Anxiety_disorder',
#             'Social/Behavioural Issues'  # <-- CORRECTED: Now uses underscore
#         ]

#         # Build the full input row for the model
#         input_data = {}
#         for col in ALL_FEATURE_COLUMNS:
#             if col in form_features:
#                 input_data[col] = int(data[col])
#             else:
#                 input_data[col] = 6 # Default value for missing features
        
#         input_df = pd.DataFrame([input_data])
#         input_df = input_df[ALL_FEATURE_COLUMNS]

#         prediction = model.predict(input_df)

#         report_text = ""
#         if prediction[0] == 'Yes':
#             report_text = """
# Analysis: Indicators Consistent with ASD Traits

# Conclusion: Based on the provided questionnaire data, the model's prediction is POSITIVE for ASD Traits...
# (Full report click here)
# """
#         else:
#             report_text = """
# Analysis: Indicators Not Consistent with ASD Traits

# Conclusion: Based on the provided questionnaire data, the model's prediction is NEGATIVE for ASD Traits...
# (Full report click here)
# """

#         return jsonify({'prediction': report_text.strip()})

#     except Exception as e:
#         traceback.print_exc()
#         return jsonify({'error': f"An error occurred: {e}"}), 400

# if __name__ == '__main__':
#     app.run(debug=True, port=5000)
#///////////////////////////////////////////////////////////////////////////////////////////////////






#///////////////////////////////////////////////////////////////////////////////////////////////////

# from flask import Flask, request, jsonify
# import pickle
# import numpy as np
# import traceback

# app = Flask(__name__)

# # Load your trained machine learning model
# try:
#     with open('model.pkl', 'rb') as model_file:
#         model = pickle.load(model_file)
#     print("✅ Model loaded successfully!")
# except Exception as e:
#     print(f"❌ Error loading model: {e}")
#     model = None

# @app.route('/predict_ml', methods=['POST'])
# def predict():
#     if model is None:
#         return jsonify({'error': 'Model is not loaded'}), 500

#     try:
#         data = request.get_json()
        
#         features = [
#             int(data['A1']), int(data['A2']), int(data['A3']),
#             int(data['A4']), int(data['A5']), int(data['A6']),
#             int(data['A7']), int(data['Anxiety_disorder']),
#             int(data['Social_Behavioural_Issues'])
#         ]

#         final_features = [np.array(features)]
#         prediction = model.predict(final_features)

#         # Generative-style static text reports
#         report_text = ""
        
#         # ✅ CORRECTED LINE: Check for the string 'Yes' instead of the number 1
#         if prediction[0] == 'Yes':
#             # This is the report for a POSITIVE prediction
#             report_text = """
# Analysis: Indicators Consistent with ASD Traits

# Conclusion: Based on the provided questionnaire data, the model's prediction is POSITIVE for ASD Traits. The combination of inputs aligns with patterns commonly associated with the autism spectrum.

# Interpretation: This result suggests that the responses regarding social interaction, communication, and behavioral patterns match the criteria the model was trained to recognize. This is a statistical indicator based on the screening data, not a medical diagnosis.

# Next Steps: It is strongly recommended to discuss these results with a pediatrician, child psychologist, or a specialist in developmental disorders. They can provide a comprehensive assessment and guide you on the appropriate next steps.
# """
#         else:
#             # This is the report for a NEGATIVE prediction
#             report_text = """
# Analysis: Indicators Not Consistent with ASD Traits

# Conclusion: Based on the provided questionnaire data, the model's prediction is NEGATIVE for ASD Traits. The combination of inputs does not align with patterns the model identifies as being commonly associated with the autism spectrum.

# Interpretation: This result suggests that the responses indicate behaviors and developmental milestones that fall within a typical range, according to the model's training data.

# Next Steps: Continue to monitor developmental milestones as a standard part of childcare. If you ever have concerns about development or behavior, it is always best to consult with a healthcare professional.
# """

#         return jsonify({'prediction': report_text.strip()})

#     except Exception as e:
#         # This block is for debugging if something else goes wrong
#         print("\n!!!!!!!!!! AN ERROR OCCURRED INSIDE THE PREDICT FUNCTION !!!!!!!!!!\n")
#         print(f"Exception Type: {type(e).__name__}")
#         print(f"Exception Details: {e}")
#         traceback.print_exc()
#         print("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
#         return jsonify({'error': f"An error occurred: {e}"}), 400

# if __name__ == '__main__':
#     app.run(debug=True, port=5000)