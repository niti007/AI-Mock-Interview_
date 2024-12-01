# AI-Mock-Interview

The AI Mock Interview Assistant is a Python application that provides personalized mock interviews based on a user's resume, job description, and the selected type of interview (technical, behavioral, or competency-based). Utilizing natural language processing, it analyzes data to conduct mock interviews, followed by a summary that highlights strengths and areas for improvement in responses.

Features
Resume and Job Description Parsing: Parses resumes (PDF or DOCX) and job descriptions to extract essential details, including skills, experience, education, and job requirements.

Interview Question Generator: Generates interview questions tailored to the selected interview type, considering information from the resume and job description to create context-relevant questions.

Answer Summary Analysis: Evaluates responses and provides feedback on technical, behavioral, competency-based, and general interview questions derived from the resume and job description.

Usage
Install Dependencies:

pip install -r requirements.txt
Run the Application:

python -m streamlit run app.py
Follow On-Screen Instructions: Upload your resume and job description, then provide interview responses to receive customized recommendations, feedback, and targeted interview questions.

Architecture
The AI Mock Interview Assistant consists of several key components:

CVParser: Extracts relevant details from resumes, including skills, experience, and education.

JobDescriptionParser: Extracts required skills, responsibilities, and qualifications from job descriptions.

ResourceRecommender: Uses parsed data and interview performance to generate personalized resource recommendations based on skill development needs and interview feedback.

QuestionGenerator: Produces tailored interview questions for technical, behavioral, and competency-based interviews.

Answer Evaluator: Assesses responses and provides detailed feedback, highlighting areas for improvement.

Streamlit UI: Provides an interactive, user-friendly interface for the mock interview process.

Contributing
Contributions are welcome! If you encounter issues or have suggestions for improvement, please submit a pull request or open an issue.

License
© 2024 AI Mock Interview Assistant. All rights reserved. Licensed under the MIT License.
