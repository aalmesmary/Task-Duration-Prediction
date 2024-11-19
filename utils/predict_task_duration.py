import pandas as pd
import joblib
import logging
import os
import math
from datetime import timedelta
import ast

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)


class TaskDurationPredictor:
    """
    A class to handle preprocessing and predicting the total task duration
    for a project schedule row.
    """

    def __init__(self, model_path: str, encoder_mapping: dict):
        """
        Initialize the predictor with the model and preprocessing requirements.

        Args:
            model_path (str): Path to the trained model file.
            encoder_mapping (dict): Mapping for label encoding categorical features.
        """
        self.model = self._load_model(model_path)
        self.encoder_mapping = encoder_mapping

    def _load_model(self, model_path: str):
        """
        Load the trained model.

        Args:
            model_path (str): Path to the model file.

        Returns:
            Trained model object.
        """
        try:
            # Dynamically resolve the model path
            script_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(script_dir, model_path)
            model = joblib.load(model_path)
            logging.info(f"Model loaded successfully from {model_path}")
            return model
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            raise

    def preprocess_row(self, row: dict) -> pd.DataFrame:
        """
        Preprocess a single row of input data for prediction.

        Args:
            row (dict): Raw input data as a dictionary.

        Returns:
            pd.DataFrame: Preprocessed data ready for prediction.
        """
        try:
            # Preprocess a single row for prediction
            # Label encode categorical features
            row["Environment"] = self.encoder_mapping.get(row["Environment"], -1)

            # Add number of dependencies
            row["Number of Dependencies"] = (
                len(row["Dependencies"]) if isinstance(row["Dependencies"], list) else 0
            )

            # Convert to DataFrame for model input
            processed_row = pd.DataFrame([row])

            # Drop unnecessary columns for model training
            processed_row = processed_row.drop(
                columns=[
                    "Task ID",
                    "Dependencies",
                    "Delay (Days)",
                    "Task Duration (Days)",
                    "Total Task Duration",
                    "Start Date",
                    "End Date",
                    "Actual End Date",
                ]
            )
            logging.info("Row successfully preprocessed")
            return processed_row

        except Exception as e:
            logging.error(f"Error during preprocessing: {e}")
            raise

    def predict(self, preprocessed_row: pd.DataFrame) -> float:
        """
        Predict the total task duration using the preprocessed input.

        Args:
            preprocessed_row (pd.DataFrame): Preprocessed input data.

        Returns:
            float: Predicted total task duration.
        """
        try:
            prediction = self.model.predict(preprocessed_row)
            logging.info("Prediction successfully made")
            return math.ceil(prediction[0])
        except Exception as e:
            logging.error(f"Error during prediction: {e}")
            raise

    def process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process the DataFrame to make predictions, handle dependencies,
        and add new columns for expected and predicted dates.

        Args:
            df (pd.DataFrame): The input DataFrame containing task data.

        Returns:
            pd.DataFrame: The updated DataFrame with predictions and adjusted dates.
        """
        df = df.copy()  # Avoid mutating the original DataFrame
        
        # Convert 'Dependencies' column from string to list, if necessary
        if df['Dependencies'].dtype == 'object':
            df['Dependencies'] = df['Dependencies'].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith('[') else []
            )

        # Add Total Task Duration
        df["Total Task Duration"] = df["Task Duration (Days)"] + df["Delay (Days)"]

        # Convert 'Start Date' and 'Actual End Date' to datetime
        df["Start Date"] = pd.to_datetime(df["Start Date"], format="%d/%m/%Y")
        df["Actual End Date"] = pd.to_datetime(df["Actual End Date"], format="%d/%m/%Y")

        # Dictionary to store predicted end dates for dependency tracking
        task_end_dates = {}
        data = []

        for idx, row in df.iterrows():
            # Preprocess row
            preprocessed_row = self.preprocess_row(row.to_dict())

            # Predict task duration
            prediction = self.predict(preprocessed_row)

            # Handle dependencies
            dependencies = row["Dependencies"]
            start_date = row["Start Date"]

            if isinstance(dependencies, list) and dependencies:
                latest_dependency_end = max(
                    [task_end_dates[dep] for dep in dependencies if dep in task_end_dates],
                    default=start_date,
                )
                start_date = max(start_date, latest_dependency_end)

            # Calculate the predicted end date
            predicted_end_date = start_date + timedelta(days=prediction)
            
            # Update the task_end_dates dictionary for dependency tracking
            task_end_dates[row['Task ID']] = predicted_end_date

            data.append({
                    "Task ID": row["Task ID"],
                    "Team Size": row["Team Size"],
                    "Resource Availability": row["Resource Availability"],
                    "Complexity": row["Complexity"],
                    "Priority": row["Priority"],
                    "Risk": row["Risk"],
                    "Environment": row["Environment"],
                    "Dependencies": row["Dependencies"],
                    "Expected Total Task Duration": row["Task Duration (Days)"] + row["Delay (Days)"],
                    "Predicted Total Task Duration": prediction,
                    "Expected Start Date": row["Start Date"],
                    "Expected End Date": row["Actual End Date"],
                    "Predicted Start Date": start_date,
                    "Predicted End Date": predicted_end_date
                }
            )
        logging.info("Successfully processed all tasks")
        return pd.DataFrame(data)


# Example Usage
if __name__ == "__main__":
    # Path to the trained model
    MODEL_PATH = "../models/best_model.pkl"

    # Encoder mapping for 'Environment'
    ENVIRONMENT_MAPPING = {"Arctic": 0, "Desert": 1, "Onshore": 2, "Offshore": 3}

    # Example row of input data
    # Example DataFrame
    script_dir = os.path.dirname(os.path.abspath(__file__))
    test_df_path = os.path.join(script_dir, "../data/Sample.csv")
    test_df = pd.read_csv(test_df_path)

    # Initialize the predictor
    predictor = TaskDurationPredictor(MODEL_PATH, ENVIRONMENT_MAPPING)

    # Process the DataFrame
    updated_df = predictor.process_dataframe(test_df)

    # Display the updated DataFrame
    print(updated_df)
    updated_df.to_csv("result.csv", index=False)
