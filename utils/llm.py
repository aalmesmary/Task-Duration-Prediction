from .config import client

class ReportGenerator:
    def __init__(self):
        self.client = client

        # Define the system prompt for the AI model
        self.system_prompt = """You are a highly skilled project management consultant specializing in 
            analyzing task schedules and dependencies. Your task is to generate a 
            professional report based on the provided DataFrame. The report should 
            Analyze and write only one insight for all tasks FOCUSE ON the schedules and delays only.
            respond must include Key Insight and conclusion only."""

    def generate_report(self, df):
        """
        Generate a professional report for schedule insights based on the provided DataFrame.

        Args:
            df (pd.DataFrame): DataFrame containing task data.

        Returns:
            str: Generated report.
        """
        # Convert the DataFrame to a JSON-like structure for input to GPT
        tasks_data = df.to_dict(orient="records")

        # Create the user prompt to pass the data
        user_prompt = f"Dataset:\n{tasks_data}"

        try:
            # Make a request to GPT
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
            )
            # Extract the AI's response
            report = response.choices[0].message.content
            return report

        except Exception as e:
            print(f"Error generating report: {e}")
            return "An error occurred while generating the report. Please try again."


# Example Usage:
# api_key = "your_openai_api_key"
# report_generator = ReportGenerator(api_key)
# report = report_generator.generate_report(df)
# print(report)
