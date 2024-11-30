import spacy
from typing import Dict, List, Optional
import re
from pathlib import Path
import logging


class JobDescriptionParser:
    def __init__(self):
        """Initialize the JD parser with spaCy model for NER and keyword extraction"""
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logging.error("Required spaCy model not found. Installing 'en_core_web_sm'...")
            from spacy.cli import download
            download("en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")

        # Common skills and requirements patterns
        self.skill_patterns = [
            r"proficient in (.*?)\.",
            r"experience with (.*?)\.",
            r"knowledge of (.*?)\.",
            r"familiarity with (.*?)\."
        ]

    def parse_job_description(self, text: str) -> Dict[str, List[str]]:
        """
        Parse job description text to extract key information

        Args:
            text (str): Raw job description text

        Returns:
            Dict with keys: required_skills, responsibilities, qualifications
        """
        parsed_data = {
            "required_skills": [],
            "responsibilities": [],
            "qualifications": [],
            "years_experience": None,
            "job_title": None
        }

        # Process text with spaCy
        doc = self.nlp(text)

        # Extract job title (usually appears early in the text)
        for ent in doc.ents:
            if ent.label_ == "WORK_OF_ART" or ent.label_ == "ORG":
                parsed_data["job_title"] = ent.text
                break

        # Extract years of experience
        experience_pattern = r"(\d+)[\+]?\s+years?(?:\s+of)?\s+experience"
        experience_match = re.search(experience_pattern, text, re.IGNORECASE)
        if experience_match:
            parsed_data["years_experience"] = int(experience_match.group(1))

        # Extract skills using patterns
        for pattern in self.skill_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                skill = match.group(1).strip()
                if skill and skill not in parsed_data["required_skills"]:
                    parsed_data["required_skills"].append(skill)

        # Extract responsibilities (usually in bullet points or numbered lists)
        responsibilities_pattern = r"(?:^|\n)[\s•*-]+(.+?)(?=\n|$)"
        responsibilities = re.finditer(responsibilities_pattern, text, re.MULTILINE)
        for resp in responsibilities:
            resp_text = resp.group(1).strip()
            if resp_text and len(resp_text) > 10:  # Filter out very short items
                parsed_data["responsibilities"].append(resp_text)

        # Extract qualifications (usually follows "requirements" or "qualifications")
        quals_section = re.split(r"(?i)requirements:|qualifications:", text)
        if len(quals_section) > 1:
            quals_text = quals_section[1]
            qualifications = re.findall(r"(?:^|\n)[\s•*-]+(.+?)(?=\n|$)", quals_text, re.MULTILINE)
            parsed_data["qualifications"].extend([q.strip() for q in qualifications if q.strip()])

        return parsed_data

    def generate_interview_focus(self, parsed_data: Dict[str, List[str]]) -> Dict[str, float]:
        """
        Generate focus areas for interview based on parsed JD

        Returns:
            Dict mapping topic areas to their importance weight (0-1)
        """
        focus_areas = {
            "technical_skills": 0.0,
            "domain_knowledge": 0.0,
            "soft_skills": 0.0,
            "experience": 0.0
        }

        # Weight technical skills based on requirements
        tech_keywords = ["programming", "software", "technical", "code", "development"]
        focus_areas["technical_skills"] = sum(
            1 for skill in parsed_data["required_skills"]
            if any(keyword in skill.lower() for keyword in tech_keywords)
        ) / max(len(parsed_data["required_skills"]), 1)

        # Weight domain knowledge
        domain_keywords = ["industry", "business", "domain", "field", "sector"]
        focus_areas["domain_knowledge"] = sum(
            1 for qual in parsed_data["qualifications"]
            if any(keyword in qual.lower() for keyword in domain_keywords)
        ) / max(len(parsed_data["qualifications"]), 1)

        # Weight soft skills
        soft_keywords = ["communication", "team", "leadership", "collaborative", "interpersonal"]
        focus_areas["soft_skills"] = sum(
            1 for resp in parsed_data["responsibilities"]
            if any(keyword in resp.lower() for keyword in soft_keywords)
        ) / max(len(parsed_data["responsibilities"]), 1)

        # Weight experience based on years required
        if parsed_data["years_experience"]:
            focus_areas["experience"] = min(parsed_data["years_experience"] / 10, 1.0)

        # Normalize weights
        total_weight = sum(focus_areas.values())
        if total_weight > 0:
            focus_areas = {k: v / total_weight for k, v in focus_areas.items()}

        return focus_areas

    def save_parsed_data(self, parsed_data: Dict[str, List[str]], output_path: Path) -> None:
        """Save parsed JD data to file"""
        output_path.write_text(str(parsed_data))