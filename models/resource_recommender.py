from langchain_groq import ChatGroq
from typing import Dict, List, Union
import json
from pathlib import Path
import os
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class ResourceRecommender:
    def __init__(self):
        # Load resource database
        self.resources = self._load_resources()
        self.groq = self._initialize_groq()

    def _initialize_groq(self):
        """Initialize the Groq LLM for dynamic resource recommendations."""
        api_key = os.getenv('GROQ_API_KEY')
        if not api_key:
            raise ValueError("Groq API key not found in environment variables.")

        try:
            groq = ChatGroq(groq_api_key=api_key, model_name="llama-3.2-90b-text-preview")
            logger.info("Successfully initialized Groq LLM for resource recommendations.")
            return groq
        except Exception as e:
            logger.error(f"Failed to initialize Groq LLM: {str(e)}")
            raise

    def _load_resources(self) -> Dict[str, List[Dict[str, str]]]:
        """Load resource database from JSON file or use default resources."""
        # Define default resources if file doesn't exist
        default_resources = {
            "python": [
                {
                    "title": "Python Documentation",
                    "url": "https://docs.python.org/3/",
                    "type": "documentation",
                    "level": "all"
                },
                # Additional resources...
            ],
            # Additional categories...
        }

        try:
            resources_path = Path("resources.json")
            if resources_path.exists():
                with open(resources_path, 'r') as f:
                    return json.load(f)
            else:
                with open(resources_path, 'w') as f:
                    json.dump(default_resources, f, indent=4)
                return default_resources
        except Exception as e:
            logger.error(f"Error loading resources: {str(e)}")
            return default_resources

    def get_recommendations(self,
                            cv_data: Dict,
                            interview_type: str,
                            interview_feedback: Dict,
                            technical_stack: List[str] = None) -> Dict[str, List[Dict[str, str]]]:
        """
        Generate personalized resource recommendations based on CV and interview performance.

        Args:
            cv_data: Parsed CV data
            interview_type: Type of interview (technical/behavioral)
            interview_feedback: Performance feedback from interview
            technical_stack: List of technical skills (for technical interviews)

        Returns:
            Dict containing recommended resources by category
        """
        recommendations = {
            "priority": [],
            "skill_development": [],
            "interview_prep": [],
            "additional": []
        }

        # Identify weak areas based on interview feedback
        weak_areas = self._identify_weak_areas(interview_feedback)

        # Get recommendations from Groq based on feedback and technical stack
        groq_recommendations = self._get_groq_recommendations(weak_areas, technical_stack)
        recommendations.update(groq_recommendations)

        return recommendations

    def _get_groq_recommendations(self, weak_areas: List[str], technical_stack: List[str]) -> Dict[str, List[Dict[str, str]]]:
        """Use Groq to generate dynamic resource recommendations based on weak areas and tech stack."""
        prompt = f"""
You are an expert career advisor providing resource recommendations based on interview performance.

The candidate's weak areas are: {', '.join(weak_areas)}.
The candidate's technical stack includes: {', '.join(technical_stack)}.

Please suggest detailed and categorized resources (priority, skill development, interview prep, additional) 
for improving skills in these weak areas. Provide recommendations in JSON format:
{{
    "priority": [{{"title": "Title", "url": "Resource URL"}}],
    "skill_development": [{{"title": "Title", "url": "Resource URL"}}],
    "interview_prep": [{{"title": "Title", "url": "Resource URL"}}],
    "additional": [{{"title": "Title", "url": "Resource URL"}}]
}}
"""
        try:
            response = self.groq.invoke(prompt)
            return json.loads(response.content.strip())
        except Exception as e:
            logger.error(f"Error generating Groq recommendations: {str(e)}")
            return {
                "priority": [],
                "skill_development": [],
                "interview_prep": [],
                "additional": []
            }

    def _identify_weak_areas(self, feedback: Dict) -> List[str]:
        """Identify areas needing improvement based on interview feedback."""
        weak_areas = []
        for question, metrics in feedback.items():
            if isinstance(metrics, dict):
                if metrics.get('clarity', 1.0) < 0.7:
                    weak_areas.append('communication')
                if metrics.get('technical_accuracy', 1.0) < 0.7:
                    weak_areas.append('technical_knowledge')
                if metrics.get('structure', 1.0) < 0.7:
                    weak_areas.append('response_structure')
                if metrics.get('problem_solving', 1.0) < 0.7:
                    weak_areas.append('problem_solving')
        return list(set(weak_areas))  # Remove duplicates