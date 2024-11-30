import spacy
import pdfplumber
import docx
import logging
from typing import Dict, Any, Union

# Initialize the spacy model once (you may change this model based on your language needs)
nlp = spacy.load("en_core_web_sm")

# Set up logging configuration
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class CVParser:
    def __init__(self):
        logger.info("CVParser initialized")

    def load_file(self, file_path: str) -> str:
        """
        Load the file and extract text. Handles PDF and DOCX formats.
        """
        logger.debug(f"Loading file: {file_path}")
        try:
            if file_path.endswith(".pdf"):
                return self._extract_text_from_pdf(file_path)
            elif file_path.endswith(".docx"):
                return self._extract_text_from_docx(file_path)
            else:
                logger.error(f"Unsupported file format for file: {file_path}")
                raise ValueError("Unsupported file format. Only PDF and DOCX are supported.")
        except Exception as e:
            logger.exception(f"Error loading file {file_path}: {e}")
            return ""

    def _extract_text_from_pdf(self, file_path: str) -> str:
        """
        Extract text from a PDF file.
        """
        logger.debug(f"Extracting text from PDF: {file_path}")
        text = ""
        try:
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            logger.info("PDF text extraction completed.")
        except Exception as e:
            logger.exception(f"Error extracting text from PDF {file_path}: {e}")
        return text

    def _extract_text_from_docx(self, file_path: str) -> str:
        """
        Extract text from a DOCX file.
        """
        logger.debug(f"Extracting text from DOCX: {file_path}")
        text = ""
        try:
            doc = docx.Document(file_path)
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            logger.info("DOCX text extraction completed.")
        except Exception as e:
            logger.exception(f"Error extracting text from DOCX {file_path}: {e}")
        return text

    def parse_cv(self, file_path: str) -> Dict[str, Any]:
        """
        Main method to parse CV sections and extract useful information.
        """
        logger.debug(f"Starting to parse CV: {file_path}")
        cv_data = {
            "experience": [],
            "education": [],
            "skills": []
        }

        # Load text from the file
        text = self.load_file(file_path)
        if not text:
            logger.error("No text found in the CV. Parsing aborted.")
            return cv_data

        # Process text with spaCy NLP
        logger.debug("Processing text with spaCy NLP")
        doc = nlp(text)

        # Extract sections
        cv_data["experience"] = self._extract_experience(doc)
        cv_data["education"] = self._extract_education(doc)
        cv_data["skills"] = self._extract_skills(doc)

        logger.info("CV parsing completed successfully")
        return cv_data

    def _extract_experience(self, doc) -> list:
        """
        Extract experience details from the parsed CV.
        """
        logger.debug("Extracting experience section")
        experience = []
        for sent in doc.sents:
            if "experience" in sent.text.lower():
                experience.append(sent.text)
        logger.debug(f"Extracted experience: {experience}")
        return experience

    def _extract_education(self, doc) -> list:
        """
        Extract education details from the parsed CV.
        """
        logger.debug("Extracting education section")
        education = []
        for sent in doc.sents:
            if "education" in sent.text.lower():
                education.append(sent.text)
        logger.debug(f"Extracted education: {education}")
        return education

    def _extract_skills(self, doc) -> list:
        """
        Extract skills details from the parsed CV.
        """
        logger.debug("Extracting skills section")
        skills = []
        for token in doc:
            if token.ent_type_ == "SKILL":  # Customize if spacy SKILL entity is configured
                skills.append(token.text)
        logger.debug(f"Extracted skills: {skills}")
        return skills


# Example usage
if __name__ == "__main__":
    parser = CVParser()
    parsed_data = parser.parse_cv("sample_cv.pdf")
    print(parsed_data)