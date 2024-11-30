from langchain_groq import ChatGroq
from typing import Dict, List, Optional
import streamlit as st
import os
from dotenv import load_dotenv
import logging
import json
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class InterviewAnswerEvaluator:
    def __init__(self):
        """Initialize the InterviewAnswerEvaluator with Groq LLM"""
        logger.info("Initializing InterviewAnswerEvaluator")
        self.api_key = os.getenv('GROQ_API_KEY')
        if not self.api_key:
            logger.error("Groq API key not found in environment variables")
            raise ValueError("Groq API key is required. Set GROQ_API_KEY in .env file")

        try:
            self.llm = ChatGroq(
                groq_api_key=self.api_key,
                model_name="llama-3.2-90b-text-preview"
            )
            logger.info("Successfully initialized Groq LLM")
        except Exception as e:
            logger.error(f"Failed to initialize Groq LLM: {str(e)}")
            raise

        # Initialize session state for storing answers and evaluations
        if 'interview_session' not in st.session_state:
            self.initialize_session_state()

    def initialize_session_state(self):
        """Initialize or reset the session state"""
        st.session_state.interview_session = {
            'answers': [],
            'current_question': 0,
            'evaluations': [],
            'start_time': datetime.now().isoformat(),
            'feedback_displayed': False
        }

    def _evaluate_answer_with_groq(self, prompt: str) -> str:
        """Evaluate the answer using Groq LLM synchronously"""
        logger.debug(f"Evaluating answer with prompt length: {len(prompt)}")
        try:
            response = self.llm.invoke(prompt)
            logger.debug(f"Received response from Groq: {response.content[:100]}...")
            return response.content.strip()
        except Exception as e:
            logger.error(f"Error in _evaluate_answer_with_groq: {str(e)}")
            return "Error in evaluation"

    def _safe_float_conversion(self, value) -> float:
        """Safely convert a value to float, returning 0 if conversion fails"""
        try:
            if isinstance(value, (int, float)):
                return float(value)
            elif isinstance(value, str):
                cleaned_value = ''.join(c for c in value if c.isdigit() or c == '.')
                return float(cleaned_value) if cleaned_value else 0.0
            return 0.0
        except (ValueError, TypeError):
            return 0.0

    def evaluate_answer(self, answer: str, question: str, cv_context: str, jd_context: str) -> Dict[str, Optional[str]]:
        """Evaluate a single answer and store in session state"""
        prompt = f"""
You are an experienced technical interviewer providing detailed feedback for each interview answer.

Question: {question}
Candidate's Answer: {answer}
Context:
- CV Summary: {cv_context}
- Job Description: {jd_context}

Provide a comprehensive evaluation in JSON format with:
{{
    "question_number": {len(st.session_state.interview_session['answers']) + 1},
    "relevance_score": "Score 0-100 for how well the answer addresses the question",
    "technical_accuracy": "Detailed assessment of technical accuracy and depth",
    "communication_clarity": "Evaluation of how clearly the answer was presented",
    "key_strengths": ["Specific", "strong", "points", "in", "answer"],
    "improvement_areas": ["Specific", "areas", "needing", "work"],
    "detailed_feedback": "Comprehensive feedback with specific examples from the answer",
    "suggestions": "Concrete suggestions for improvement",
    "overall_score": "Numerical score 0-100",
    "quick_tips": ["Short", "actionable", "improvement", "tips"]
}}
"""
        try:
            evaluation_json_str = self._evaluate_answer_with_groq(prompt)
            evaluation = json.loads(evaluation_json_str)

            # Store answer and evaluation
            answer_data = {
                'question': question,
                'answer': answer,
                'evaluation': evaluation,
                'timestamp': datetime.now().isoformat()
            }

            st.session_state.interview_session['answers'].append(answer_data)
            return evaluation

        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {str(e)}")
            return self._get_default_evaluation()

    def display_answer_feedback(self, index: int) -> None:
        """Display feedback for a specific answer"""
        try:
            answer_data = st.session_state.interview_session['answers'][index]
            eval_data = answer_data['evaluation']

            st.subheader(f"Question {index + 1} Feedback")

            # Create columns for layout
            col1, col2 = st.columns([2, 1])

            with col1:
                st.markdown("#### Question")
                st.write(answer_data['question'])

                st.markdown("#### Your Answer")
                st.write(answer_data['answer'])

                st.markdown("#### Detailed Feedback")
                st.write(eval_data.get('detailed_feedback', 'No detailed feedback available'))

                st.markdown("#### Technical Assessment")
                st.write(eval_data.get('technical_accuracy', 'No technical assessment available'))

            with col2:
                # Display scores
                st.metric("Overall Score", f"{self._safe_float_conversion(eval_data.get('overall_score', 0)):.1f}/100")
                st.metric("Relevance Score",
                          f"{self._safe_float_conversion(eval_data.get('relevance_score', 0)):.1f}/100")

                # Display strengths and improvements
                st.markdown("#### Key Strengths")
                for strength in eval_data.get('key_strengths', []):
                    st.markdown(f"- {strength}")

                st.markdown("#### Areas for Improvement")
                for area in eval_data.get('improvement_areas', []):
                    st.markdown(f"- {area}")

            # Display quick tips in an expander
            with st.expander("Quick Tips for Improvement"):
                for tip in eval_data.get('quick_tips', []):
                    st.markdown(f"- {tip}")

        except Exception as e:
            st.error(f"Error displaying feedback: {str(e)}")

    def display_all_feedback(self) -> None:
        """Display feedback for all answers"""
        if not st.session_state.interview_session['answers']:
            st.warning("No answers to evaluate yet.")
            return

        st.title("Interview Feedback Summary")

        # Overall statistics
        total_questions = len(st.session_state.interview_session['answers'])
        average_score = sum(
            self._safe_float_conversion(answer['evaluation'].get('overall_score', 0))
            for answer in st.session_state.interview_session['answers']
        ) / total_questions if total_questions > 0 else 0

        # Display overall metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Questions", total_questions)
        with col2:
            st.metric("Average Score", f"{average_score:.1f}/100")
        with col3:
            st.metric("Completion Time", f"{len(st.session_state.interview_session['answers'])} mins")

        # Display individual feedback for each answer
        for i in range(total_questions):
            st.markdown("---")
            self.display_answer_feedback(i)

    def _get_default_evaluation(self) -> Dict[str, any]:
        """Return default evaluation structure when parsing fails"""
        return {
            'question_number': len(st.session_state.interview_session['answers']) + 1,
            'relevance_score': 0,
            'technical_accuracy': "Error evaluating technical accuracy",
            'communication_clarity': "Error evaluating communication clarity",
            'key_strengths': [],
            'improvement_areas': [],
            'detailed_feedback': "Error generating detailed feedback",
            'suggestions': "Error generating suggestions",
            'overall_score': 0,
            'quick_tips': []
        }

    def clear_session(self):
        """Clear the current interview session"""
        self.initialize_session_state()
        logger.info("Cleared interview session")