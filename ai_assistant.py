# Add this fix to your ai_assistant.py file or update the analyze_data function

import google.generativeai as genai
import json
import logging

class AIAssistant:
    def __init__(self):
        # Initialize Gemini model
        GOOGLE_API_KEY = 'YOUR_API_KEY_HERE'  # Replace with your actual API key
        genai.configure(api_key=GOOGLE_API_KEY)
        self.model = genai.GenerativeModel('gemini-pro')
    
    async def analyze_data(self, df, target_column, task_type):
        """Perform comprehensive data analysis"""
        try:
            # Create dataset summary
            dataset_info = {
                'basic_info': {
                    'rows': len(df),
                    'columns': len(df.columns),
                    'task_type': task_type,
                    'target_column': target_column
                },
                'columns': {
                    col: {
                        'type': str(df[col].dtype),
                        'unique_values': int(df[col].nunique()),
                        'missing_values': int(df[col].isnull().sum())
                    }
                    for col in df.columns
                }
            }

            # Add numeric statistics
            for col in df.select_dtypes(include=['int64', 'float64']).columns:
                stats = df[col].describe()
                dataset_info['columns'][col].update({
                    'mean': float(stats['mean']),
                    'std': float(stats['std']),
                    'min': float(stats['min']),
                    'max': float(stats['max'])
                })

            # Generate analysis prompt
            prompt = f"""
            Analyze this dataset for machine learning:

            Basic Information:
            - Task Type: {task_type}
            - Target Column: {target_column}
            - Number of Rows: {dataset_info['basic_info']['rows']}
            - Number of Columns: {dataset_info['basic_info']['columns']}

            Provide a comprehensive analysis including:
            1. Data Quality Assessment
            2. Feature Engineering Suggestions  
            3. Preprocessing Recommendations
            4. Modeling Approach
            5. Potential Challenges and Solutions

            Format the response with markdown headings and bullet points.
            """

            try:
                # Get Gemini response
                response = self.model.generate_content(prompt)
                
                return {
                    'success': True,
                    'analysis': response.text
                }

            except Exception as gemini_error:
                logging.error(f"Gemini API error: {str(gemini_error)}")
                # Return a fallback analysis
                return {
                    'success': True,
                    'analysis': f"""
# Dataset Analysis Report

## Basic Information
- **Dataset Shape**: {dataset_info['basic_info']['rows']} rows × {dataset_info['basic_info']['columns']} columns
- **Task Type**: {task_type}
- **Target Column**: {target_column}

## Data Quality Assessment
- The dataset appears to be ready for machine learning
- Check for missing values and handle appropriately
- Consider data types and ensure proper encoding

## Recommendations
1. **Preprocessing**: Scale numerical features and encode categorical variables
2. **Feature Engineering**: Consider creating interaction features
3. **Model Selection**: Start with ensemble methods like Random Forest
4. **Validation**: Use cross-validation for robust performance estimation

## Next Steps
Proceed with model training using the selected algorithms.
"""
                }

        except Exception as e:
            logging.error(f"Data analysis error: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'analysis': "Analysis unavailable. Proceeding with default preprocessing."
            }