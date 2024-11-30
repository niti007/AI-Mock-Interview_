import streamlit as st
import os
import tempfile
from utils.cv_parser import CVParser
from utils.JD_parser import JobDescriptionParser
from models.question_generator import QuestionGenerator
from models.resource_recommender import ResourceRecommender
from models.answer_evaluator import InterviewAnswerEvaluator
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)


def load_api_key():
    """Load API key from environment variables"""
    api_key = os.getenv('GROQ_API_KEY')
    if not api_key:
        st.error("Groq API key not found. Please set GROQ_API_KEY in your .env file")
        st.stop()
    return api_key


def save_uploadedfile(uploadedfile):
    """Save uploaded file to a temporary file and return the path"""
    if uploadedfile is None:
        return None
    try:
        suffix = os.path.splitext(uploadedfile.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(uploadedfile.getvalue())
            return tmp_file.name
    except Exception as e:
        logger.error(f"Error saving file: {str(e)}", exc_info=True)
        st.error(f"Error saving file: {str(e)}")
        return None


def initialize_session_state():
    """Initialize all session state variables"""
    session_vars = {
        'current_step': 'upload',
        'interview_data': {},
        'interview_progress': 0,
        'current_question': 0,
        'questions': [],
        'responses': [],
        'feedback': {},
        'temp_files': [],
        'question_generator': None,
        'answer_evaluator': None,
        'current_answer': '',
        'interview_session': {'answers': []},
        'answer_submitted': False  # New flag to track answer submission
    }

    for var, default in session_vars.items():
        if var not in st.session_state:
            st.session_state[var] = default
    logger.debug("Session state initialized with variables: %s", st.session_state)


def cleanup_temp_files():
    """Clean up temporary files when the session ends"""
    for temp_file in st.session_state.temp_files:
        try:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
                logger.debug(f"Cleaned up temporary file: {temp_file}")
        except Exception as e:
            logger.warning(f"Error cleaning up temporary file {temp_file}: {str(e)}")


def process_documents(cv_temp_path, jd_temp_path, cv_parser, jd_parser):
    """Process the uploaded CV and JD files"""
    try:
        logger.debug("Starting document processing")
        cv_data = cv_parser.parse_cv(cv_temp_path)
        logger.debug(f"CV parsing result: {cv_data}")
        st.session_state.interview_data['cv_data'] = cv_data
        if cv_temp_path:
            st.session_state['persistent_cv_data'] = cv_data  # Store CV data persistently

        if jd_temp_path:
            jd_data = jd_parser.parse_job_description(jd_temp_path)
            logger.debug(f"JD parsing result: {jd_data}")
            st.session_state.interview_data['jd_data'] = jd_data

        return True
    except Exception as e:
        logger.error(f"Error processing documents: {str(e)}", exc_info=True)
        st.error(f"Error processing documents: {str(e)}")
        return False


def clear_answer():
    """Clear the current answer and reset submission flag"""
    st.session_state.current_answer = ''
    st.session_state.answer_submitted = False


def next_question():
    """Advance to the next question"""
    st.session_state.current_question += 1
    st.session_state.answer_submitted = False
    st.session_state.current_answer = ''


def display_interview_summary():
    """Display the interview summary and feedback"""
    st.header("Interview Summary")

    if st.session_state.feedback:
        # Calculate metrics based on the feedback structure
        metrics = {
            'relevance': sum(float(f.get('relevance', '0').rstrip('%')) / 100
                             for f in st.session_state.feedback.values()) / len(st.session_state.feedback),
            'clarity': sum(float(f.get('clarity', '0').rstrip('%')) / 100
                           for f in st.session_state.feedback.values()) / len(st.session_state.feedback),
            'skills_demonstration': sum(float(f.get('skills_demonstration', '0').rstrip('%')) / 100
                                        for f in st.session_state.feedback.values()) / len(st.session_state.feedback)
        }

        # Display metrics
        cols = st.columns(3)
        for col, (metric, value) in zip(cols, metrics.items()):
            col.metric(f"Average {metric.title()}", f"{value:.0%}")

        # Display detailed feedback
        st.subheader("Detailed Feedback")
        for idx, response in enumerate(st.session_state.responses):
            with st.expander(f"Question {idx + 1}"):
                st.write("**Question:**", response['question'])
                st.write("**Your Response:**", response['response'])
                feedback = st.session_state.feedback.get(idx, {})

                # Display each feedback component
                st.write("**Relevance:**", feedback.get('relevance', 'No feedback available.'))
                st.write("**Clarity:**", feedback.get('clarity', 'No feedback available.'))
                st.write("**Skills Demonstration:**", feedback.get('skills_demonstration', 'No feedback available.'))
                st.write("**Alignment:**", feedback.get('alignment', 'No feedback available.'))

        # Display recommended resources
        resource_recommender = ResourceRecommender()
        try:
            resources = resource_recommender.recommend_resources(
                metrics['clarity'],
                metrics['relevance'],
                metrics['skills_demonstration']
            )
            if resources:
                st.subheader("Recommended Resources")
                for resource in resources:
                    st.write(resource)
        except Exception as e:
            logger.error(f"Failed to recommend resources: {str(e)}", exc_info=True)
            st.error("An error occurred while fetching recommended resources. Please try again later.")


def main():
    st.set_page_config(
        page_title="AI Mock Interview Assistant",
        page_icon="🎯",
        layout="wide"
    )

    initialize_session_state()

    # Initialize QuestionGenerator and AnswerEvaluator if not already initialized
    if st.session_state.question_generator is None:
        api_key = load_api_key()
        try:
            st.session_state.question_generator = QuestionGenerator()
            st.session_state.answer_evaluator = InterviewAnswerEvaluator()
            logger.info("QuestionGenerator and AnswerEvaluator initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize QuestionGenerator or AnswerEvaluator: {str(e)}", exc_info=True)
            st.error(f"Failed to initialize interview system: {str(e)}")
            st.stop()

    # Add model info to sidebar
    with st.sidebar:
        st.title("Interview Setup")
        st.markdown("*Using Llama 3.2 90B model*")
        interview_type = st.selectbox(
            "Select Interview Type",
            ["Competency Based", "Behavioral", "Technical"],
            help="Choose the type of interview you want to practice"
        )

        if interview_type == "Technical":
            technical_stack = st.multiselect(
                "Select Technical Stack",
                ["Python", "JavaScript", "Java", "React", "Node.js", "SQL"],
                help="Choose technologies you want to be interviewed on"
            )
            st.session_state.interview_data['technical_stack'] = technical_stack

    st.title("AI Mock Interview Assistant")

    # Upload Step
    if st.session_state.current_step == 'upload':
        st.header("Upload Documents")

        with st.expander("📌 Tips for better results", expanded=True):
            st.markdown("""
            - Ensure your CV is up-to-date
            - Include relevant skills and experience
            - For technical interviews, highlight your project experience
            """)

        cv_file = st.file_uploader("Upload your CV (Required)", type=["pdf", "docx"])
        jd_file = st.file_uploader("Upload Job Description (Optional)", type=["pdf", "docx", "txt"])

        if cv_file is not None:
            cv_parser = CVParser()
            jd_parser = JobDescriptionParser()

            if st.button("Process Documents", type="primary"):
                with st.spinner("Processing your documents..."):
                    cv_temp_path = save_uploadedfile(cv_file)
                    if cv_temp_path:
                        st.session_state.temp_files.append(cv_temp_path)
                        jd_temp_path = save_uploadedfile(jd_file) if jd_file else None
                        if jd_temp_path:
                            st.session_state.temp_files.append(jd_temp_path)

                        if process_documents(cv_temp_path, jd_temp_path, cv_parser, jd_parser):
                            st.session_state.current_step = 'confirm'
                            st.success("Documents processed successfully!")
                            st.rerun()

    # Confirmation Step
    elif st.session_state.current_step == 'confirm':
        st.header("Confirm Interview Setup")

        st.subheader("Parsed Information")
        st.write("Interview Type:", interview_type)
        if 'technical_stack' in st.session_state.interview_data:
            st.write("Technical Stack:", ", ".join(st.session_state.interview_data['technical_stack']))

        if st.button("Start Interview", type="primary"):
            with st.spinner("Generating interview questions..."):
                try:
                    logger.debug("Starting question generation with parameters:")
                    logger.debug(f"Interview type: {interview_type}")
                    logger.debug(f"CV data: {st.session_state.interview_data.get('cv_data')}")
                    logger.debug(f"JD data: {st.session_state.interview_data.get('jd_data')}")
                    logger.debug(f"Technical stack: {st.session_state.interview_data.get('technical_stack')}")

                    questions = st.session_state.question_generator.generate_questions(
                        question_type=interview_type.lower().replace(" ", "_"),
                        resume_info=st.session_state.interview_data.get('cv_data'),
                        job_description=st.session_state.interview_data.get('jd_data'),
                        technical_stack=st.session_state.interview_data.get('technical_stack')
                    )

                    logger.debug(f"Generated questions: {questions}")

                    if questions and len(questions) > 0:
                        st.session_state.questions = questions
                        st.session_state.current_step = 'interview'
                        st.rerun()
                    else:
                        logger.error("Question generation returned empty list")
                        st.error("Failed to generate interview questions. Please try again.")
                except Exception as e:
                    logger.error(f"Error generating questions: {str(e)}", exc_info=True)
                    st.error(f"Error generating questions: {str(e)}")

    # Interview Step
    elif st.session_state.current_step == 'interview':
        st.header(f"Interview - {interview_type}")

        # Display progress
        total_questions = len(st.session_state.questions)
        current_progress = (st.session_state.current_question / total_questions) * 100
        st.progress(current_progress / 100)
        st.write(f"Question {st.session_state.current_question + 1} of {total_questions}")

        if st.session_state.current_question < total_questions:
            # Display current question
            question = st.session_state.questions[st.session_state.current_question]
            st.write(f"**Question:** {question}")

            # Text area for answer
            user_response = st.text_area(
                "Your Answer",
                key=f"answer_{st.session_state.current_question}",
                height=200
            )

            # Submit button
            if st.button("Submit Answer", key=f"submit_{st.session_state.current_question}"):
                if user_response.strip():  # Check if answer is not empty
                    # Get CV and JD context from session state
                    cv_context = st.session_state.interview_data.get('cv_data', '')
                    jd_context = st.session_state.interview_data.get('jd_data', '')

                    # Evaluate answer
                    feedback = st.session_state.answer_evaluator.evaluate_answer(
                        answer=user_response,
                        question=question,
                        cv_context=cv_context,
                        jd_context=jd_context
                    )

                    # Store feedback and response
                    st.session_state.feedback[st.session_state.current_question] = feedback
                    st.session_state.responses.append({
                        "question": question,
                        "response": user_response
                    })

                    # Move to next question or summary
                    if st.session_state.current_question + 1 >= total_questions:
                        st.session_state.current_step = 'summary'
                    else:
                        st.session_state.current_question += 1

                    st.rerun()
                else:
                    st.warning("Please provide an answer before submitting.")

        else:
            st.session_state.current_step = 'summary'
            st.rerun()

    # Summary Step
    elif st.session_state.current_step == 'summary':
        display_interview_summary()

        # Add option to start new interview with modified reset
        if st.button("Start New Interview"):
            # Store CV data temporarily
            temp_cv_data = st.session_state.get('persistent_cv_data', None)
            # Reset session state
            for key in list(st.session_state.keys()):
                if key not in ['question_generator', 'answer_evaluator']:
                    del st.session_state[key]
            initialize_session_state()
            # Restore CV data
            if temp_cv_data:
                st.session_state.interview_data['cv_data'] = temp_cv_data
                st.session_state.current_step = 'confirm'  # Skip upload step
            st.rerun()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}", exc_info=True)
        st.error(f"An unexpected error occurred: {str(e)}")
        cleanup_temp_files()