from langchain_groq import ChatGroq
from typing import List, Dict, Optional
from enum import Enum
import streamlit as st
import os
from dotenv import load_dotenv
import logging
import re
import random
import time

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('question_generator.log')
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class QuestionType(Enum):
    TECHNICAL = "technical"
    BEHAVIORAL = "behavioral"
    COMPETENCY = "competency_based"

class QuestionGenerator:
    def __init__(self):
        """Initialize the QuestionGenerator with Groq LLM"""
        logger.info("Initializing QuestionGenerator")
        self.api_key = os.getenv('GROQ_API_KEY')
        if not self.api_key:
            logger.error("Groq API key not found in environment variables")
            raise ValueError("Groq API key is required. Set GROQ_API_KEY in .env file")

        try:
            self.llm = ChatGroq(
                groq_api_key=self.api_key,
                model_name="llama-3.1-70b-versatile",
                temperature=0.7
            )
            logger.info("Successfully initialized Groq LLM")
        except Exception as e:
            logger.error(f"Failed to initialize Groq LLM: {str(e)}")
            raise

        # Define prompts for different question types
        self.type_prompts = {
            QuestionType.TECHNICAL: """
You are an experienced technical interviewer. Generate exactly 15 UNIQUE and DIFFERENT technical interview questions.
Ensure these questions are different from previous attempts.

Technical Skills Required: {technical_stack}
Candidate Background: {cv_context}
Job Requirements: {jd_context}

Additional rules for variety:
1. Mix different difficulty levels
2. Include both theoretical and practical questions
3. Vary the format (multiple choice, open-ended, scenario-based)
4. Cover different aspects of each technology
5. Include some unexpected but relevant questions

[Previous prompts may have generated similar questions, ensure these are completely different]

Format the output exactly as follows:
1. First question here
2. Second question here
[...and so on until 15 questions]""",

            QuestionType.BEHAVIORAL: """
You are an experienced HR interviewer. Generate exactly 15 UNIQUE and DIFFERENT behavioral interview questions based on the following context:

Candidate Background: {cv_context}
Job Requirements: {jd_context}

Rules for generating questions:
1. Questions should follow the STAR format
2. Focus on past experiences and specific situations
3. Include questions about teamwork and leadership
4. Cover conflict resolution and problem-solving
5. Make questions relevant to the candidate's experience level
6. Include questions about handling pressure and deadlines
7. Add questions about adapting to change
8. Include questions about diversity and inclusion
9. Cover remote work and collaboration scenarios
10. Add questions about mentoring and knowledge sharing

Format the output exactly as follows:
1. First question here
2. Second question here
[...and so on until 15 questions]""",

            QuestionType.COMPETENCY: """
You are an experienced competency-based interviewer. Generate exactly 15 UNIQUE and DIFFERENT competency-based questions using the following context:

Candidate Background: {cv_context}
Job Requirements: {jd_context}

Rules for generating questions:
1. Focus on specific skills and competencies required for the role
2. Include questions about project management and delivery
3. Cover communication and stakeholder management
4. Address decision-making and problem-solving abilities
5. Make questions measurable and evidence-based
6. Include questions about innovation and continuous improvement
7. Add questions about strategic thinking and planning
8. Cover leadership and influence
9. Include questions about resource management
10. Address cross-functional collaboration

Format the output exactly as follows:
1. First question here
2. Second question here
[...and so on until 15 questions]"""
        }
        logger.info("Question templates initialized")

        # Initialize a set to track previously generated questions
        self.previous_questions = set()
        self.max_history = 1000  # Maximum number of questions to remember

    def _get_dynamic_temperature(self):
        """Generate a random temperature for more variability"""
        return random.uniform(0.7, 0.9)

    def _get_random_seed(self):
        """Generate a random seed based on timestamp"""
        return int(time.time() * 1000)

    def _prepare_context(self, resume_info: Optional[Dict], job_description: Optional[Dict], technical_stack: Optional[List[str]] = None) -> Dict:
        """Prepare context for question generation"""
        logger.debug("Preparing context for question generation")
        logger.debug(f"Resume info: {resume_info}")
        logger.debug(f"Job description: {job_description}")
        logger.debug(f"Technical stack: {technical_stack}")

        # Format CV context
        cv_context = "Not provided"
        if resume_info:
            cv_parts = []
            if 'skills' in resume_info:
                cv_parts.append(f"Skills: {', '.join(resume_info['skills'])}")
            if 'experience' in resume_info:
                cv_parts.append(f"Experience: {resume_info['experience']}")
            if 'education' in resume_info:
                cv_parts.append(f"Education: {resume_info['education']}")
            cv_context = ". ".join(cv_parts)

        # Format JD context
        jd_context = "Not provided"
        if job_description:
            jd_parts = []
            if 'requirements' in job_description:
                jd_parts.append(f"Requirements: {', '.join(job_description['requirements'])}")
            if 'responsibilities' in job_description:
                jd_parts.append(f"Responsibilities: {', '.join(job_description['responsibilities'])}")
            jd_context = ". ".join(jd_parts)

        # Format technical stack
        tech_stack_str = ", ".join(technical_stack) if technical_stack else "General technical skills"

        context = {
            "cv_context": cv_context,
            "jd_context": jd_context,
            "technical_stack": tech_stack_str
        }

        logger.debug(f"Prepared context: {context}")
        return context

    def _is_question_unique(self, question: str) -> bool:
        """Check if a question is unique"""
        return question not in self.previous_questions

    def _add_to_history(self, question: str):
        """Add question to history"""
        self.previous_questions.add(question)
        if len(self.previous_questions) > self.max_history:
            self.previous_questions.pop()

    def _generate_with_groq(self, prompt: str) -> List[str]:
        """Generate questions using Groq LLM with dynamic parameters"""
        logger.debug(f"Generating questions with prompt length: {len(prompt)}")
        try:
            # Set dynamic temperature
            temperature = self._get_dynamic_temperature()
            # Add timestamp and random seed to prompt
            seed = self._get_random_seed()
            enhanced_prompt = f"""[Seed: {seed}]
            Generate unique and different questions for this attempt.
            {prompt}"""

            # Update LLM parameters
            self.llm = ChatGroq(
                groq_api_key=self.api_key,
                model_name="llama-3.1-70b-versatile",
                temperature=temperature
            )

            messages = [{"role": "user", "content": enhanced_prompt}]
            response = self.llm.invoke(messages)

            # Log the raw response for debugging
            logger.debug(f"Raw Groq response: {response.content}")

            questions = []
            # Split response into lines and process each line
            for line in response.content.strip().split('\n'):
                line = line.strip()
                # Improved number detection regex
                if line and re.match(r'^\d{1,2}[\.$$]', line):
                    try:
                        # Extract question after the number and any delimiter
                        question = re.split(r'^\d{1,2}[\.$$]\s*', line)[1].strip()
                        if self._is_question_unique(question):
                            questions.append(question)
                            self._add_to_history(question)
                        else:
                            logger.debug(f"Duplicate question found: {question}")
                            continue
                    except IndexError:
                        logger.warning(f"Failed to parse line: {line}")
                        continue

            logger.info(f"Generated {len(questions)} questions successfully")

            # If we got fewer than 15 questions, log a warning
            if len(questions) < 15:
                logger.warning(f"Only generated {len(questions)} questions, expected 15")

            # Ensure we always return exactly 15 questions
            while len(questions) < 15:
                default_q = f"Default question #{len(questions) + 1}"
                questions.append(default_q)
                logger.warning(f"Added default question: {default_q}")

            return questions[:15]

        except Exception as e:
            logger.error(f"Error in _generate_with_groq: {str(e)}", exc_info=True)
            st.error(f"Error generating questions with Groq: {str(e)}")
            # Return default questions on error
            default_questions = [f"Default question {i + 1}" for i in range(15)]
            return default_questions

    def generate_questions(self, question_type: str, resume_info: Optional[Dict] = None, job_description: Optional[Dict] = None, technical_stack: Optional[List[str]] = None) -> List[str]:
        """Generate interview questions based on type and context with improved error handling"""
        logger.info(f"Generating questions of type: {question_type}")
        try:
            # Validate inputs
            if not question_type:
                raise ValueError("Question type is required")

            # Convert question type to match app's selection
            if question_type.lower() == "competency based":
                question_type = "competency_based"

            # Get question type enum
            try:
                q_type = QuestionType(question_type.lower())
            except ValueError as e:
                logger.error(f"Invalid question type: {question_type}")
                raise ValueError(f"Invalid question type: {question_type}. Must be one of {[t.value for t in QuestionType]}")

            # Log input data
            logger.debug(f"Resume info: {resume_info}")
            logger.debug(f"Job description: {job_description}")
            logger.debug(f"Technical stack: {technical_stack}")

            # Prepare context
            context = self._prepare_context(resume_info, job_description, technical_stack)

            # Get prompt template and format it
            if q_type not in self.type_prompts:
                raise ValueError(f"No prompt template found for question type: {q_type}")

            prompt = self.type_prompts[q_type].format(**context)
            logger.debug(f"Generated prompt: {prompt[:200]}...")

            # Generate questions using Groq
            questions = self._generate_with_groq(prompt)

            # Validate output
            if not questions:
                logger.error("No questions generated")
                raise ValueError("Failed to generate questions")

            return questions

        except Exception as e:
            logger.error(f"Error in generate_questions: {str(e)}", exc_info=True)
            st.error(f"Error in question generation: {str(e)}")
            default_questions = [f"Default {question_type} question {i + 1}" for i in range(15)]
            logger.info("Returning default questions due to error")
            return default_questions

def test_question_generator():
    """Test function for QuestionGenerator"""
    try:
        logger.info("Starting QuestionGenerator test")

        # Test data
        test_resume = {
            "skills": ["Python", "Machine Learning", "API Development"],
            "experience": "3 years as Software Engineer, 2 years as ML Engineer",
            "education": "MS in Computer Science"
        }

        test_job = {
            "requirements": ["Python expertise", "ML knowledge", "API design"],
            "responsibilities": ["Lead ML projects", "Design APIs"],
            "role": "Senior Software Engineer"
        }

        # Initialize generator
        generator = QuestionGenerator()

        # Test each question type
        for q_type in QuestionType:
            logger.info(f"Testing question type: {q_type.value}")
            questions = generator.generate_questions(
                question_type=q_type.value,
                resume_info=test_resume,
                job_description=test_job,
                technical_stack=["Python", "Machine Learning"] if q_type == QuestionType.TECHNICAL else None
            )

            logger.info(f"Generated {len(questions)} questions for {q_type.value}")
            for i, q in enumerate(questions, 1):
                logger.debug(f"Question {i}: {q}")

        logger.info("QuestionGenerator test completed successfully")

    except Exception as e:
        logger.error(f"Error in test_question_generator: {str(e)}", exc_info=True)

if __name__ == "__main__":
    test_question_generator()