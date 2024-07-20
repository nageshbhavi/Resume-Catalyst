import warnings
warnings.filterwarnings('ignore')

from flask import Flask, request, render_template, send_from_directory, url_for
from PyPDF2 import PdfReader
import re
import pickle

import os
from werkzeug.utils import secure_filename
from pydparser import ResumeParser

app = Flask(__name__)

# Load models===========================================================================================================
rf_classifier_categorization = pickle.load(open('models/rf_classifier_categorization.pkl', 'rb'))
tfidf_vectorizer_categorization = pickle.load(open('models/tfidf_vectorizer_categorization.pkl', 'rb'))
rf_classifier_job_recommendation = pickle.load(open('models/rf_classifier_job_recommendation.pkl', 'rb'))
tfidf_vectorizer_job_recommendation = pickle.load(open('models/tfidf_vectorizer_job_recommendation.pkl', 'rb'))

# Define the upload folder=========================================================================
UPLOAD_FOLDER = 'uploded_resumes/'  # Ensure this directory exists and is writable
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Clean resume==========================================================================================================
def cleanResume(txt):
    cleanText = re.sub(r'http\S+\s', ' ', txt)  # Fixing \S
    cleanText = re.sub(r'RT|cc', ' ', cleanText)
    cleanText = re.sub(r'#\S+\s', ' ', cleanText)  # Fixing \S
    cleanText = re.sub(r'@\S+', '  ', cleanText)  # Fixing \S
    cleanText = re.sub(r'[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"""), ' ', cleanText)  # Fixing \]
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText)
    cleanText = re.sub(r'\s+', ' ', cleanText)  # Fixing \s
    return cleanText

# Prediction and Category Name
def predict_category(resume_text):
    resume_text = cleanResume(resume_text)
    resume_tfidf = tfidf_vectorizer_categorization.transform([resume_text])
    predicted_category = rf_classifier_categorization.predict(resume_tfidf)[0]
    return predicted_category

# Prediction and Category Name
def job_recommendation(resume_text):
    resume_text= cleanResume(resume_text)
    resume_tfidf = tfidf_vectorizer_job_recommendation.transform([resume_text])
    recommended_job = rf_classifier_job_recommendation.predict(resume_tfidf)[0]
    return recommended_job

def pdf_to_text(file):
    reader = PdfReader(file)
    text = ''
    for page in range(len(reader.pages)):
        text += reader.pages[page].extract_text()
    return text




# resume parsing
import re

def extract_contact_number_from_resume(text):
    contact_number = None

    # Use regex pattern to find a potential contact number
    pattern = r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
    match = re.search(pattern, text)
    if match:
        contact_number = match.group()

    return contact_number

def extract_email_from_resume(text):
    email = None

    # Use regex pattern to find a potential email address
    pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
    match = re.search(pattern, text)
    if match:
        email = match.group()

    return email

def extract_skills_from_resume(text):
    # List of predefined skills
    skills_list = [
        'Python', 'Data Analysis', 'Machine Learning', 'Communication', 'Project Management', 'Deep Learning', 'SQL',
        'Tableau','C','C++','spaCY','NLP','nltk'
        'Java', 'C++', 'JavaScript', 'HTML', 'CSS', 'React', 'Angular', 'Node.js', 'MongoDB', 'Express.js', 'Git',
        'Research', 'Statistics', 'Quantitative Analysis', 'Qualitative Analysis', 'SPSS', 'R', 'Data Visualization',
        'Matplotlib',
        'Seaborn', 'Plotly', 'Pandas', 'Numpy', 'Scikit-learn', 'TensorFlow', 'Keras', 'PyTorch', 'NLTK', 'Text Mining',
        'Natural Language Processing', 'Computer Vision', 'Image Processing', 'OCR', 'Speech Recognition',
        'Recommendation Systems',
        'Collaborative Filtering', 'Content-Based Filtering', 'Reinforcement Learning', 'Neural Networks',
        'Convolutional Neural Networks',
        'Recurrent Neural Networks', 'Generative Adversarial Networks', 'XGBoost', 'Random Forest', 'Decision Trees',
        'Support Vector Machines',
        'Linear Regression', 'Logistic Regression', 'K-Means Clustering', 'Hierarchical Clustering', 'DBSCAN',
        'Association Rule Learning',
        'Apache Hadoop', 'Apache Spark', 'MapReduce', 'Hive', 'HBase', 'Apache Kafka', 'Data Warehousing', 'ETL',
        'Big Data Analytics',
        'Cloud Computing', 'Amazon Web Services (AWS)', 'Microsoft Azure', 'Google Cloud Platform (GCP)', 'Docker',
        'Kubernetes', 'Linux',
        'Shell Scripting', 'Cybersecurity', 'Network Security', 'Penetration Testing', 'Firewalls', 'Encryption',
        'Malware Analysis',
        'Digital Forensics', 'CI/CD', 'DevOps', 'Agile Methodology', 'Scrum', 'Kanban', 'Continuous Integration',
        'Continuous Deployment',
        'Software Development', 'Web Development', 'Mobile Development', 'Backend Development', 'Frontend Development',
        'Full-Stack Development',
        'UI/UX Design', 'Responsive Design', 'Wireframing', 'Prototyping', 'User Testing', 'Adobe Creative Suite',
        'Photoshop', 'Illustrator',
        'InDesign', 'Figma', 'Sketch', 'Zeplin', 'InVision', 'Product Management', 'Market Research',
        'Customer Development', 'Lean Startup',
        'Business Development', 'Sales', 'Marketing', 'Content Marketing', 'Social Media Marketing', 'Email Marketing',
        'SEO', 'SEM', 'PPC',
        'Google Analytics', 'Facebook Ads', 'LinkedIn Ads', 'Lead Generation', 'Customer Relationship Management (CRM)',
        'Salesforce',
        'HubSpot', 'Zendesk', 'Intercom', 'Customer Support', 'Technical Support', 'Troubleshooting',
        'Ticketing Systems', 'ServiceNow',
        'ITIL', 'Quality Assurance', 'Manual Testing', 'Automated Testing', 'Selenium', 'JUnit', 'Load Testing',
        'Performance Testing',
        'Regression Testing', 'Black Box Testing', 'White Box Testing', 'API Testing', 'Mobile Testing',
        'Usability Testing', 'Accessibility Testing',
        'Cross-Browser Testing', 'Agile Testing', 'User Acceptance Testing', 'Software Documentation',
        'Technical Writing', 'Copywriting',
        'Editing', 'Proofreading', 'Content Management Systems (CMS)', 'WordPress', 'Joomla', 'Drupal', 'Magento',
        'Shopify', 'E-commerce',
        'Payment Gateways', 'Inventory Management', 'Supply Chain Management', 'Logistics', 'Procurement',
        'ERP Systems', 'SAP', 'Oracle',
        'Microsoft Dynamics', 'Tableau', 'Power BI', 'QlikView', 'Looker', 'Data Warehousing', 'ETL',
        'Data Engineering', 'Data Governance',
        'Data Quality', 'Master Data Management', 'Predictive Analytics', 'Prescriptive Analytics',
        'Descriptive Analytics', 'Business Intelligence',
        'Dashboarding', 'Reporting', 'Data Mining', 'Web Scraping', 'API Integration', 'RESTful APIs', 'GraphQL',
        'SOAP', 'Microservices',
        'Serverless Architecture', 'Lambda Functions', 'Event-Driven Architecture', 'Message Queues', 'GraphQL',
        'Socket.io', 'WebSockets'
                     'Ruby', 'Ruby on Rails', 'PHP', 'Symfony', 'Laravel', 'CakePHP', 'Zend Framework', 'ASP.NET', 'C#',
        'VB.NET', 'ASP.NET MVC', 'Entity Framework',
        'Spring', 'Hibernate', 'Struts', 'Kotlin', 'Swift', 'Objective-C', 'iOS Development', 'Android Development',
        'Flutter', 'React Native', 'Ionic',
        'Mobile UI/UX Design', 'Material Design', 'SwiftUI', 'RxJava', 'RxSwift', 'Django', 'Flask', 'FastAPI',
        'Falcon', 'Tornado', 'WebSockets',
        'GraphQL', 'RESTful Web Services', 'SOAP', 'Microservices Architecture', 'Serverless Computing', 'AWS Lambda',
        'Google Cloud Functions',
        'Azure Functions', 'Server Administration', 'System Administration', 'Network Administration',
        'Database Administration', 'MySQL', 'PostgreSQL',
        'SQLite', 'Microsoft SQL Server', 'Oracle Database', 'NoSQL', 'MongoDB', 'Cassandra', 'Redis', 'Elasticsearch',
        'Firebase', 'Google Analytics',
        'Google Tag Manager', 'Adobe Analytics', 'Marketing Automation', 'Customer Data Platforms', 'Segment',
        'Salesforce Marketing Cloud', 'HubSpot CRM',
        'Zapier', 'IFTTT', 'Workflow Automation', 'Robotic Process Automation (RPA)', 'UI Automation',
        'Natural Language Generation (NLG)',
        'Virtual Reality (VR)', 'Augmented Reality (AR)', 'Mixed Reality (MR)', 'Unity', 'Unreal Engine', '3D Modeling',
        'Animation', 'Motion Graphics',
        'Game Design', 'Game Development', 'Level Design', 'Unity3D', 'Unreal Engine 4', 'Blender', 'Maya',
        'Adobe After Effects', 'Adobe Premiere Pro',
        'Final Cut Pro', 'Video Editing', 'Audio Editing', 'Sound Design', 'Music Production', 'Digital Marketing',
        'Content Strategy', 'Conversion Rate Optimization (CRO)',
        'A/B Testing', 'Customer Experience (CX)', 'User Experience (UX)', 'User Interface (UI)', 'Persona Development',
        'User Journey Mapping', 'Information Architecture (IA)',
        'Wireframing', 'Prototyping', 'Usability Testing', 'Accessibility Compliance', 'Internationalization (I18n)',
        'Localization (L10n)', 'Voice User Interface (VUI)',
        'Chatbots', 'Natural Language Understanding (NLU)', 'Speech Synthesis', 'Emotion Detection',
        'Sentiment Analysis', 'Image Recognition', 'Object Detection',
        'Facial Recognition', 'Gesture Recognition', 'Document Recognition', 'Fraud Detection',
        'Cyber Threat Intelligence', 'Security Information and Event Management (SIEM)',
        'Vulnerability Assessment', 'Incident Response', 'Forensic Analysis', 'Security Operations Center (SOC)',
        'Identity and Access Management (IAM)', 'Single Sign-On (SSO)',
        'Multi-Factor Authentication (MFA)', 'Blockchain', 'Cryptocurrency', 'Decentralized Finance (DeFi)',
        'Smart Contracts', 'Web3', 'Non-Fungible Tokens (NFTs)']


    skills = []

    for skill in skills_list:
        pattern = r"\b{}\b".format(re.escape(skill))
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            skills.append(skill)

    return skills


#making a model using spacy----work in progress
def extract_name_from_resume(text):
    name = None

    # Use regex pattern to find a potential name
    pattern = r"(\b[A-Z][a-z]+\b)\s(\b[A-Z][a-z]+\b)"
    match = re.search(pattern, text)
    if match:
        name = match.group()

    return name

#=========================calculating resume-score==========================================
import re

def evaluate_resume(text,name,email):
    # Calculate impact score (0-10)
    impact_score = calculate_impact(text)

    # Calculate brevity score (0-5)
    brevity_score = calculate_brevity(text)

    section_score = calculate_sections(text,name,email)

    # Combine scores (you can adjust weights if needed)
    total_score = 0.25 * impact_score + 0.25 * brevity_score + section_score
    print("TOTAL SCORE: ", total_score)
    total_score=round(total_score,2)
    print("type of total_scr:",type(total_score))
    return total_score


def calculate_impact(resume):
    # Regular expression to find numbers
    number_pattern = re.compile(r'\b\d+\b')
    
    # List of impact words
    impact_words = [
        'Accelerated', 'Achieved', 'Attained', 'Completed', 'Conceived', 'Convinced', 'Discovered', 'Doubled', 
        'Effected', 'Eliminated', 'Expanded', 'Expedited', 'Founded', 'Improved', 'Increased', 'Initiated', 
        'Innovated', 'Introduced', 'Invented', 'Launched', 'Mastered', 'Overcame', 'Overhauled', 'Pioneered', 
        'Reduced', 'Resolved', 'Revitalized', 'Spearheaded', 'Strengthened', 'Transformed', 'Upgraded', 'Tripled', 
        'Addressed', 'Advised', 'Arranged', 'Authored', 'Co-authored', 'Co-ordinated', 'Communicated', 'Corresponded', 
        'Counselled', 'Developed', 'Demonstrated', 'Directed', 'Drafted', 'Enlisted', 'Facilitated', 'Formulated', 
        'Guided', 'Influenced', 'Interpreted', 'Interviewed', 'Instructed', 'Lectured', 'Led', 'Liaised', 'Mediated', 
        'Moderated', 'Motivated', 'Negotiated', 'Persuaded', 'Presented', 'Promoted', 'Proposed', 'Publicized', 
        'Recommended', 'Reconciled', 'Recruited', 'Resolved', 'Taught', 'Trained', 'Translated', 'Composed', 
        'Conceived', 'Created', 'Designed', 'Developed', 'Devised', 'Established', 'Founded', 'Generated', 'Implemented', 
        'Initiated', 'Instituted', 'Introduced', 'Launched', 'Led', 'Opened', 'Originated', 'Pioneered', 'Planned', 
        'Prepared', 'Produced', 'Promoted', 'Started', 'Released', 'Administered', 'Analyzed', 'Assigned', 'Chaired', 
        'Consolidated', 'Contracted', 'Co-ordinated', 'Delegated', 'Developed', 'Directed', 'Evaluated', 'Executed', 
        'Organized', 'Planned', 'Prioritized', 'Produced', 'Recommended', 'Reorganized', 'Reviewed', 'Scheduled', 
        'Supervised', 'Managed', 'Guided', 'Advised', 'Coached', 'Conducted', 'Directed', 'Guided', 'Demonstrated', 
        'Illustrated', 'Led', 'Managed', 'Organized', 'Performed', 'Presented', 'Taught', 'Trained', 'Mentored', 
        'Spearheaded', 'Authored', 'Accelerated', 'Achieved', 'Allocated', 'Completed', 'Awarded', 'Persuaded', 
        'Revamped', 'Influenced', 'Assessed', 'Clarified', 'Counseled', 'Diagnosed', 'Educated', 'Facilitated', 
        'Familiarized', 'Motivated', 'Referred', 'Rehabilitated', 'Reinforced', 'Represented', 'Moderated', 'Verified', 
        'Adapted', 'Coordinated', 'Developed', 'Enabled', 'Encouraged', 'Evaluated', 'Explained', 'Informed', 'Instructed', 
        'Lectured', 'Stimulated', 'Analyzed', 'Assessed', 'Classified', 'Collated', 'Defined', 'Devised', 'Established', 
        'Evaluated', 'Forecasted', 'Identified', 'Interviewed', 'Investigated', 'Researched', 'Tested', 'Traced', 'Designed', 
        'Interpreted', 'Verified', 'Uncovered', 'Clarified', 'Collected', 'Critiqued', 'Diagnosed', 'Examined', 'Extracted', 
        'Inspected', 'Inspired', 'Organized', 'Reviewed', 'Summarized', 'Surveyed', 'Systemized', 'Arranged', 'Budgeted', 
        'Composed', 'Conceived', 'Conducted', 'Controlled', 'Co-ordinated', 'Eliminated', 'Improved', 'Investigated', 
        'Itemised', 'Modernised', 'Operated', 'Organised', 'Planned', 'Prepared', 'Processed', 'Produced', 'Redesigned', 
        'Reduced', 'Refined', 'Researched', 'Resolved', 'Reviewed', 'Revised', 'Scheduled', 'Simplified', 'Solved', 
        'Streamlined', 'Transformed', 'Examined', 'Revamped', 'Combined', 'Consolidated', 'Converted', 'Cut', 'Decreased', 
        'Developed', 'Devised', 'Doubled', 'Tripled', 'Eliminated', 'Expanded', 'Improved', 'Increased', 'Innovated', 
        'Minimised', 'Modernised', 'Recommended', 'Redesigned', 'Reduced', 'Refined', 'Reorganised', 'Resolved', 
        'Restructured', 'Revised', 'Saved', 'Serviced', 'Simplified', 'Solved', 'Streamlined', 'Strengthened', 
        'Transformed', 'Trimmed', 'Unified', 'Widened', 'Broadened', 'Revamped', 'Administered', 'Allocated', 'Analyzed', 
        'Appraised', 'Audited', 'Balanced', 'Budgeted', 'Calculated', 'Computed', 'Developed', 'Managed', 'Planned', 
        'Projected', 'Researched', 'Restructured', 'Modelled', 'Arbitrated', 'Acted', 'Conceptualized', 'Created', 
        'Customized', 'Designed', 'Developed', 'Directed', 'Redesigned', 'Established', 'Fashioned', 'Illustrated', 
        'Instituted', 'Integrated', 'Performed', 'Planned', 'Proved', 'Revised', 'Revitalized', 'Set up', 'Shaped', 
        'Streamlined', 'Structured', 'Tabulated', 'Validated', 'Acted', 'Conceptualized', 'Created', 'Customized', 
        'Designed', 'Developed', 'Directed', 'Redesigned', 'Established', 'Fashioned', 'Illustrated', 'Instituted', 
        'Integrated', 'Performed', 'Planned', 'Proved', 'Revised', 'Revitalized', 'Set up', 'Shaped', 'Streamlined', 
        'Structured', 'Tabulated', 'Validated', 'Conceptualized', 'Coded', 'Computed', 'Extrapolated', 'Predicted', 
        'Installed', 'Engineered', 'Calculated', 'Segmented', 'Restructured', 'Arbitrated', 'Estimated', 'Overhauled', 
        'Devised', 'Assembled', 'Unified', 'Visualized', 'Debugged', 'Customized', 'Standardized', 'Steered', 'Validated', 
        'Diagnosed', 'Tested', 'Automated', 'Strengthened', 'Troubleshooted', 'Architected', 'Discovered', 'Deployed', 
        'Approved', 'Arranged', 'Catalogued', 'Classified', 'Collected', 'Compiled', 'Dispatched', 'Executed', 
        'Generated', 'Implemented', 'Inspected', 'Monitored', 'Operated', 'Ordered', 'Organized', 'Prepared', 'Processed', 
        'Purchased', 'Recorded', 'Retrieved', 'Screened', 'Specified', 'Systematized'
    ]
    
    # Find all numbers in the resume
    numbers = number_pattern.findall(resume)
    
    # Count the occurrences of numbers
    count_numbers = len(numbers) - 5
    # print("numbers-count:", count_numbers)
    
    # Find and count occurrences of impact words (case insensitive)
    count_impact_words = sum(resume.lower().count(word.lower()) for word in impact_words)
    # print("impact-words-count:", count_impact_words)
    
    # Calculate number score (0-5 scale)
    number_score = min(count_numbers / 4, 5)
    
    # Calculate impact words score based on specified ranges
    if count_impact_words < 30:
        impact_words_score = 1
    elif count_impact_words <= 50:
        impact_words_score = 3
    else:
        impact_words_score = 5
    
    # Combine both scores to get overall impact score (0-10 scale)
    impact_score = number_score + impact_words_score
    print("IMPACT:", impact_score)
    impact_score = round(impact_score,2)
    
    return impact_score


def calculate_brevity(resume):
    # Calculate the number of words in the resume
    word_count = len(resume.split())
    # print("word-count : ", word_count)
    
    # Regular expression to find bullet points (common ones: '*', '-', '•')
    bullet_point_pattern = re.compile(r'[-•*]')

    # Find all bullet points in the resume
    bullet_points = bullet_point_pattern.findall(resume)

    # Number of bullet points
    bullet_point_count = len(bullet_points)
    # print("Bullete-count : ", bullet_point_count)

    # Calculate brevity score (Example: 5 for less than 100 words, 0 for more than 500 words)
    if word_count > 1000:
        brevity_score = 3
    elif word_count > 800:
        brevity_score = 2
    elif word_count > 600:
        brevity_score = 1.5
    elif word_count > 400:
        brevity_score = 1
    elif word_count > 300:
        brevity_score = 0.5
    else:
        brevity_score = 0
    
    # Adjust brevity score based on bullet points
    if bullet_point_count > 25:
        brevity_score = max(brevity_score + 2, 0) 
    elif bullet_point_count < 25:
        brevity_score = max(brevity_score + 1, 0) 

    print("BREVITY : ",brevity_score*2)
    brevity_score = round(brevity_score,2)
    return brevity_score*2

# Declare global variables for section scores
global_contact_score = 0
global_skills_score = 0
global_projects_score = 0
global_experience_score = 0
global_other_sections_score = 0  # New global variable for other sections

def calculate_sections(resume, name=None, email=None, phone=None):
    global global_contact_score, global_skills_score, global_projects_score
    global global_experience_score, global_other_sections_score
    
    # List of sections to check
    sections_to_check = ['skills', 'projects', 'extracurricular', 'achievements', 'responsibilities']
    
    # Initialize scores for each section
    section_scores = {'contact': 0, 'skills': 0, 'projects': 0, 'experience': 0, 'extracurricular': 0, 'achievements': 0, 'responsibilities': 0}
    
    # Check if contact information (name, email, mobile) is present
    if name:
        section_scores['contact'] = 0.5

    if email:
        section_scores['contact'] += 0.5
    
    # Update the global contact score
    global_contact_score = section_scores['contact']
    
    if any(keyword in resume.lower() for keyword in ['work experience', 'experience', 'experiences']) and \
       any(keyword in resume.lower() for keyword in ['worked', 'intern', 'internship']):
        section_scores['experience'] = 1

    
    global_experience_score = section_scores['experience']
    # Check presence of other sections
    for section in sections_to_check:
        if section in resume.lower():
            section_scores[section] = 1
            
            # Update the corresponding global variable
            if section == 'skills':
                global_skills_score = section_scores[section]
            elif section == 'projects':
                global_projects_score = section_scores[section]

    
    # Calculate score for the other three combined sections
    global_other_sections_score = sum(section_scores[section] for section in ['extracurricular', 'achievements', 'responsibilities']) / 3
    
    # Cap the score at 1
    global_other_sections_score = min(global_other_sections_score, 1)
    
    # Combine scores
    total_sec_score = sum(section_scores.values()) - section_scores['extracurricular'] - section_scores['achievements'] - section_scores['responsibilities'] + global_other_sections_score
    total_sec_score = round(total_sec_score,2)
    return total_sec_score

# evaluate_resume(text)
####================gemini-api===========================
import google.generativeai as genai
import os

# Set the API key as an environment variable
os.environ['GOOGLE_API_KEY'] = 'PROVIDE_YOUR_GOOGLE_API_KEY'

# Configure the generative AI library with the API key
genai.configure(api_key=os.environ['GOOGLE_API_KEY'])

import markdown

def recommendation(skills, job_role, prompt):
    model = genai.GenerativeModel('gemini-1.5-pro-latest')
    
    # Combine skills, job role, and prompt into a single string or list of strings
    content_text = f"Skills: {', '.join(skills)}. Job Role: {job_role}. {prompt}"
    
    # Print the content_text to verify its structure
    print("Final Prompt (content_text):", content_text)
    
    # Construct the parameters dictionary with the 'text' key
    parameters = {
        "text": content_text
    }
    
    # Print the parameters to verify their structure
    print("Parameters being passed:", parameters)
    
    try:
        # Attempt to call the generate_content method with the adjusted parameters
        response = model.generate_content(parameters)
        markdown_text = response.text
        html_content = markdown.markdown(markdown_text)
        return html_content
    except TypeError as e:
        # Print the error message and the method signature for debugging
        print("Error:", e)
        import inspect
        print("Method signature:", inspect.signature(model.generate_content))
        return None



# routes===============================================

@app.route('/')
def resume():
    # Provide a simple UI to upload a resume
    return render_template("resume.html")


@app.route('/pred', methods=['POST'])
def pred():
    if 'resume' not in request.files:
        return render_template('resume.html', message="No resume file uploaded.")

    file = request.files['resume']
    if file.filename == '':
        return render_template('resume.html', message="No selected file")

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    if filename.endswith('.pdf') or filename.endswith('.txt'):
        file.save(file_path)

        if filename.endswith('.pdf'):
            text = pdf_to_text(file_path)  # Assuming you have a function to extract text from a PDF
        elif filename.endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()

        data = ResumeParser(file_path).get_extracted_data()  # Pass the file path to ResumeParser
        predicted_category = predict_category(text)
        recommended_job = job_recommendation(text)

        phone = data.get('mobile_number')
        if not phone:
            phone = extract_contact_number_from_resume(text)

        email = data.get('email')
        if not email:
            email = extract_email_from_resume(text)

        extracted_skills = extract_skills_from_resume(text)
        name = data.get('name')
        if not name:
            name = extract_name_from_resume(text)

        # ======RESUME-SCORE-PART======================================================================
        impact_score = calculate_impact(text)
        brevity_score = calculate_brevity(text)
        section_score = calculate_sections(text, name, email)
        total_score = 0.25 * impact_score + 0.25 * brevity_score + section_score


        
        total_score = round(total_score,2)
        # print("type of total_scr:",type(total_score))

        # Prepare feedback messages
        impact_message = []
        if impact_score < 6:
            impact_message.append( "You have less number of action verbs and sentences in your resume.") 
            impact_message.append("For example, weaker: Responsible for improving productivity among workers")
            impact_message.append("stronger: Conducted workload assessments and devised new operational processes that led to a 40% increase in productivity.")
            impact_message.append("We suggest you to add more numbers (e.g., increased by 40%) and more action verbs to improve the impact of your resume.")
        else:
            impact_message.append("You have an excellent impact score for your resume and meet majority of our requirements.") 
            impact_message.append("You have used more action verbs and numbers (e.g., increased by 40%) in your resume.")

        brevity_message = []
        if brevity_score < 5:
            brevity_message.append("You have a low number of bullet points and words in your resume.")
            brevity_message.append("We suggest you to add more bullet points to improve the brevity score of your resume.")
        else:
            brevity_message.append("You have an excellent brevity score for your resume and meet majority of our requirements.")
            brevity_message.append("You have used more bullet points in your resume.")

        # Section score messages
        section_messages = []
        if global_contact_score:
            section_messages.append("Excellent! You have added required contact details.")
        else:
            section_messages.append("Please add contact details like name, email, contact no. to your resume.")

        if global_skills_score:
            section_messages.append("Excellent! You have mentioned your skills in your resume.")
        else:
            section_messages.append("Please add your skills to your resume.")

        if global_projects_score:
            section_messages.append("Excellent! You have added your projects in your resume.")
        else:
            section_messages.append("Please add your projects to your resume.")

        if global_experience_score:
            section_messages.append("Excellent! You have mentioned your experience in professional fields in your resume.")
        else:
            section_messages.append("If you have any work experience, mention it in your resume.")

        # Total score messages
        total_message = []
        if total_score <= 4.5:
            total_message.append("You have a poor resume score.")
            total_message.append("Please consider the above suggestions to improve your resume score.")
        elif 4.5 < total_score <= 7:
            total_message.append("You have a good resume score but still need improvement.") 
            total_message.append("Please consider the above suggestions to improve your resume score.")
        else:
            total_message.append("You have a great resume score and have a good chance of selection.") 
            total_message.append("Your resume meets majority of our requirements.")

        # Define the skills, job role, and prompt
        skills = extracted_skills
        job_role = recommended_job
        prompt = (
            """as an HR with 13 years of experience, provide a separate list of additional 5 specific skills and no soft skills that are beneficial for this role, 
            and another separate list of 5 recommended courses with links to acquire these skills. The format should be Skills:(5 skills )  and course : (5 course )extra information is strictly not needed"""
        )

        # Call the recommendation function and store the result
        # global result
        result = recommendation(skills, job_role, prompt)

        return render_template('resume.html', filename=filename, predicted_category=predicted_category,
                               recommended_job=recommended_job, phone=phone, name=name,
                               email=email, extracted_skills=extracted_skills, impact_score=impact_score * 10,
                               brevity_score=brevity_score * 10, section_score=section_score * 10 * 2,
                               total_score=total_score * 10,
                               impact_message=impact_message, brevity_message=brevity_message,
                               section_messages=section_messages, total_message=total_message, result=result)
    else:
        return render_template('resume.html', message="Invalid file format. Please upload a PDF or TXT file.")

@app.route('/uploads/<filename>')
def download_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == '__main__':
    app.run(debug=True)


