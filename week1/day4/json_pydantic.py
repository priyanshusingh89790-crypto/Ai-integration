
import os
import json

from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

from pydantic import BaseModel, Field

from pypdf import PdfReader


# ============================================================
# 1. ENVIRONMENT / GROQ SETUP
# ============================================================

load_dotenv()

my_api_key = os.getenv("Grok_api_key")

if not my_api_key:
    raise ValueError("Groq API key not found in .env")

client = Groq(api_key=my_api_key)

model = "openai/gpt-oss-120b"


# ============================================================
# 2. JOB DESCRIPTION
# ============================================================

job_description = """
About the job

At Aspect, we are one of London's largest property maintenance teams,
covering more trades than anyone else and operating 24/7. After more than
25 years of trusted service to thousands of residential and commercial
customers, we are entering an exciting phase of rapid digital transformation.

We are investing heavily in AI and automation to reimagine how we deliver
customer service, field operations, and internal workflows.

We are now looking for a talented Full Stack Engineer with React expertise
to help us build the products and platforms that will run at the core of
our business.

The Role

As a Full Stack Engineer, you will design and ship polished,
production-grade web applications end to end.

You will spend most of your time in React building fast, accessible,
and beautiful interfaces, while also owning the APIs, data models,
and services behind them.

Your work will power everything from customer-facing booking and service
tools to the internal platforms our operations teams rely on every day.

You will partner closely with Product, Design, and our AI engineers to
turn ideas into real, shippable software.

What You Will Be Doing

Build and ship production React applications with TypeScript, focusing on
performance, accessibility, and a clean user experience.

Own features end to end, from the front-end interface through to the APIs,
data models, and services that support them.

Design and build backend services in Node.js, along with REST and GraphQL
APIs that connect our products.

Model and work with relational data in PostgreSQL, keeping schemas clean
and queries efficient.

Collaborate with Product and Design to turn concepts into intuitive,
well-crafted interfaces.

Work alongside our AI engineers to integrate LLM and agentic features
into products in a natural way.

Contribute to front-end architecture, component libraries, and engineering
standards as we scale.

Write clean, well-tested, maintainable code and participate in code reviews
to keep quality high across the team.

What We Are Looking For

Strong commercial experience building production web applications with
React and TypeScript.

Solid full-stack ability, comfortable owning both the front end and
the services behind them.

Experience with Node.js and building REST or GraphQL APIs.

Good working knowledge of relational databases, ideally PostgreSQL.

A strong eye for design detail, usability, and accessibility.

Experience with cloud platforms such as AWS, Google Cloud, or Azure,
and with CI/CD, containers, and deployment.

A track record of owning features end to end and shipping reliably.

Excellent communication skills and the ability to work closely with
non-technical stakeholders.

Preferred Qualifications

Experience with Next.js and Tailwind CSS.

Experience integrating LLM or GenAI features into web applications.

Familiarity with Salesforce, WordPress, or other CMS and CRM platforms.

Experience with Python, useful for working alongside our AI stack.

Exposure to vector databases, real-time features, or event-driven
architectures.

Experience in property maintenance, home services, or customer service
platforms is a bonus.
"""


# ============================================================
# 3. JOB DESCRIPTION SCHEMA
# ============================================================

class JobDescription(BaseModel):

    role: str = ""

    required_skills: list[str] = Field(default_factory=list)

    preferred_skills: list[str] = Field(default_factory=list)

    minimum_exp: str = ""

    qualifications: list[str] = Field(default_factory=list)

    responsibilities: list[str] = Field(default_factory=list)


# ============================================================
# 4. EXPERIENCE SCHEMA
# ============================================================

class Experience(BaseModel):

    company: str = ""

    role: str = ""

    duration: str = ""

    description: str = ""

    skills_used: list[str] = Field(default_factory=list)


# ============================================================
# 5. RESUME SCHEMA
# ============================================================

class Resume(BaseModel):

    name: str = ""

    email: str = ""

    phone: str = ""

    total_experience_years: float = 0.0

    skills: list[str] = Field(default_factory=list)

    experience: list[Experience] = Field(default_factory=list)

    education: list[str] = Field(default_factory=list)

    projects: list[str] = Field(default_factory=list)

    certifications: list[str] = Field(default_factory=list)


# ============================================================
# 6. FINAL MATCH RESULT SCHEMA
# ============================================================

class MatchResult(BaseModel):

    overall_score: float

    recommendation: str

    summary: str

    matched_required_skills: list[str] = Field(default_factory=list)

    missing_required_skills: list[str] = Field(default_factory=list)

    matched_preferred_skills: list[str] = Field(default_factory=list)

    missing_preferred_skills: list[str] = Field(default_factory=list)

    experience_match_score: float

    technical_skill_score: float

    preferred_skill_score: float

    strengths: list[str] = Field(default_factory=list)

    weaknesses: list[str] = Field(default_factory=list)

    why_good_fit: list[str] = Field(default_factory=list)

    concerns: list[str] = Field(default_factory=list)


# ============================================================
# 7. FIND ALL RESUME PDF FILES
# ============================================================

def find_resume_pdfs(folder: str = "resumes") -> list[Path]:

    resume_folder = Path(folder)

    if not resume_folder.exists():
        raise FileNotFoundError(
            f"Resume folder not found: {resume_folder}"
        )

    pdf_files = list(resume_folder.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF resumes found inside: {resume_folder}"
        )

    return pdf_files


# ============================================================
# 8. EXTRACT TEXT FROM PDF
# ============================================================

def extract_resume_text(pdf_path: Path) -> str:

    reader = PdfReader(str(pdf_path))

    text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    if not text.strip():

        raise ValueError(
            f"No text could be extracted from {pdf_path.name}. "
            "The PDF may be scanned/image-based."
        )

    return text


# ============================================================
# 9. PARSE JOB DESCRIPTION
# ============================================================

def parse_job_description(
    job_description_text: str
) -> JobDescription:

    job_schema = JobDescription.model_json_schema()

    system_prompt = f"""
You are an expert technical recruiter.

Your task is to analyze a job description and extract structured
information from it.

Return ONLY valid JSON.

Use exactly this schema:

{json.dumps(job_schema, indent=2)}

Rules:

- Extract only information actually present in the job description.
- Do not invent information.
- required_skills should contain explicitly required technical
  and professional skills.
- preferred_skills should contain preferred, nice-to-have,
  or bonus skills.
- minimum_exp should contain the minimum required experience
  if mentioned.
- qualifications should contain important qualifications.
- responsibilities should contain the main responsibilities.
- Keep skills as separate list items.
- Keep responsibilities as separate list items.
"""

    user_prompt = f"""
Analyze the following job description:

{job_description_text}
"""

    response = client.chat.completions.create(
        model=model,

        messages=[
            {
                "role": "system",
                "content": system_prompt
            },

            {
                "role": "user",
                "content": user_prompt
            }
        ],

        response_format={
            "type": "json_object"
        }
    )

    raw_json = response.choices[0].message.content

    if not raw_json:

        raise ValueError(
            "LLM returned an empty job description response"
        )

    data = json.loads(raw_json)

    return JobDescription(**data)


# ============================================================
# 10. PARSE RESUME
# ============================================================

def parse_resume(resume_text: str) -> Resume:

    resume_schema = Resume.model_json_schema()

    system_prompt = f"""
You are an expert technical recruiter.

Your task is to extract structured information from a candidate resume.

Return ONLY valid JSON.

Use exactly this schema:

{json.dumps(resume_schema, indent=2)}

Rules:

- Extract only information actually present in the resume.
- Do not invent any information.
- If information is missing, use an empty string, empty list,
  or 0.
- Keep every employment experience as a separate object.
- Put technologies explicitly mentioned for each job
  into skills_used.
- Extract projects separately from employment experience.
- Extract education separately.
- Extract certifications separately.
- Calculate total_experience_years only when it can reasonably
  be determined from the resume.
- Do not assume experience that is not supported by the resume.
"""

    user_prompt = f"""
Extract structured information from this resume:

{resume_text}
"""

    response = client.chat.completions.create(
        model=model,

        messages=[
            {
                "role": "system",
                "content": system_prompt
            },

            {
                "role": "user",
                "content": user_prompt
            }
        ],

        response_format={
            "type": "json_object"
        }
    )

    raw_json = response.choices[0].message.content

    if not raw_json:

        raise ValueError(
            "LLM returned an empty resume response"
        )

    data = json.loads(raw_json)

    return Resume(**data)


# ============================================================
# 11. COMPARE JOB DESCRIPTION WITH RESUME
# ============================================================

def final_score(
    job: JobDescription,
    resume: Resume
) -> MatchResult:

    match_schema = MatchResult.model_json_schema()

    system_prompt = f"""
You are an expert technical recruiter and candidate evaluation assistant.

Your task is to compare a structured job description with a structured
candidate resume.

Evaluate how well the candidate fits the job.

Return ONLY valid JSON.

Use exactly this schema:

{json.dumps(match_schema, indent=2)}


SCORING:

overall_score:

90-100 = Excellent Fit
75-89  = Strong Fit
60-74  = Moderate Fit
40-59  = Weak Fit
0-39   = Poor Fit


technical_skill_score:

Evaluate how closely the candidate's demonstrated technical skills
match the required technical skills.


preferred_skill_score:

Evaluate how closely the candidate matches preferred and bonus skills.


experience_match_score:

Consider:

- Total professional experience
- Relevant professional experience
- Similarity of responsibilities
- Production experience
- Full-stack experience
- Ownership of features
- Relevant projects


IMPORTANT MATCHING RULES:

- Only give credit for skills demonstrated in the resume.
- Do not invent experience.
- Do not assume related technologies are the same.

For example:

React does NOT automatically mean Next.js.

JavaScript does NOT automatically mean TypeScript.

Node.js does NOT automatically mean GraphQL.

SQL does NOT automatically mean PostgreSQL.

Python does NOT automatically mean AI/LLM experience.

Cloud experience should not automatically mean AWS.

Docker should not automatically mean Kubernetes.

Only give direct skill credit when evidence exists.


However, related skills can be mentioned as partial/transferable
experience in the explanation.


matched_required_skills:

Only include required skills clearly demonstrated in the resume.


missing_required_skills:

Include important required skills that are not demonstrated.


matched_preferred_skills:

Only include preferred skills clearly demonstrated.


missing_preferred_skills:

Include preferred skills that are not demonstrated.


strengths:

Explain the candidate's strongest areas for this role.


weaknesses:

Explain important skill or experience gaps.


why_good_fit:

Give concrete evidence-based reasons why the candidate is suitable.


concerns:

Mention hiring concerns, missing evidence, or things that should
be verified in an interview.


Be objective and evidence-based.
"""


    user_prompt = f"""
Compare the following candidate against the job description.


==============================
JOB DESCRIPTION
==============================

{job.model_dump_json(indent=2)}


==============================
CANDIDATE RESUME
==============================

{resume.model_dump_json(indent=2)}


==============================
TASK
==============================

Evaluate this candidate for the position.

Return the final structured match result.
"""


    response = client.chat.completions.create(
        model=model,

        messages=[
            {
                "role": "system",
                "content": system_prompt
            },

            {
                "role": "user",
                "content": user_prompt
            }
        ],

        response_format={
            "type": "json_object"
        }
    )


    raw_json = response.choices[0].message.content

    if not raw_json:

        raise ValueError(
            "LLM returned an empty match response"
        )

    data = json.loads(raw_json)

    return MatchResult(**data)


# ============================================================
# 12. PROCESS ONE RESUME
# ============================================================

def process_resume(
    pdf_path: Path,
    job: JobDescription
):

    print("\n========================================")
    print(f"PROCESSING: {pdf_path.name}")
    print("========================================")

    # Extract PDF text

    print("\nExtracting PDF text...")

    resume_text = extract_resume_text(pdf_path)

    print(
        f"Extracted {len(resume_text)} characters"
    )


    # Convert resume text to structured Resume

    print("\nStructuring resume...")

    resume = parse_resume(resume_text)


    # Compare resume with JD

    print("\nCalculating candidate match...")

    result = final_score(job, resume)


    return resume, result


# ============================================================
# 13. MAIN
# ============================================================

def main():

    print("PROGRAM STARTED")


    # --------------------------------------------------------
    # STEP 1
    # --------------------------------------------------------

    print("\n========================================")
    print("STEP 1: ANALYZING JOB DESCRIPTION")
    print("========================================")

    job = parse_job_description(job_description)

    print(
        job.model_dump_json(indent=2)
    )


    # --------------------------------------------------------
    # STEP 2
    # --------------------------------------------------------

    print("\n========================================")
    print("STEP 2: FINDING RESUMES")
    print("========================================")

    pdf_files = find_resume_pdfs(".")
    print(
        f"Found {len(pdf_files)} resume(s)"
    )

    for pdf in pdf_files:

        print(
            f"  - {pdf.name}"
        )


    # --------------------------------------------------------
    # STEP 3
    # --------------------------------------------------------

    print("\n========================================")
    print("STEP 3: PROCESSING CANDIDATES")
    print("========================================")


    candidates = []


    for pdf_path in pdf_files:

        try:

            resume, result = process_resume(
                pdf_path,
                job
            )


            candidate = {

                "resume_file": pdf_path.name,

                "name": resume.name,

                "email": resume.email,

                "phone": resume.phone,

                "total_experience_years":
                    resume.total_experience_years,

                "score": result.overall_score,

                "recommendation":
                    result.recommendation,

                "summary":
                    result.summary,

                "matched_required_skills":
                    result.matched_required_skills,

                "missing_required_skills":
                    result.missing_required_skills,

                "matched_preferred_skills":
                    result.matched_preferred_skills,

                "missing_preferred_skills":
                    result.missing_preferred_skills,

                "experience_match_score":
                    result.experience_match_score,

                "technical_skill_score":
                    result.technical_skill_score,

                "preferred_skill_score":
                    result.preferred_skill_score,

                "strengths":
                    result.strengths,

                "weaknesses":
                    result.weaknesses,

                "why_good_fit":
                    result.why_good_fit,

                "concerns":
                    result.concerns,

                "resume":
                    resume.model_dump()

            }


            candidates.append(candidate)


        except Exception as error:

            print(
                f"\nERROR processing {pdf_path.name}:"
            )

            print(error)


    # --------------------------------------------------------
    # STEP 4
    # --------------------------------------------------------

    print("\n========================================")
    print("STEP 4: RANKING CANDIDATES")
    print("========================================")


    candidates.sort(
        key=lambda candidate: candidate["score"],
        reverse=True
    )


    # Add ranking

    for index, candidate in enumerate(
        candidates,
        start=1
    ):

        candidate["rank"] = index


    # --------------------------------------------------------
    # STEP 5
    # --------------------------------------------------------

    print("\n========================================")
    print("FINAL CANDIDATE RANKING")
    print("========================================")


    if not candidates:

        print(
            "No candidates were successfully processed."
        )

        return


    for candidate in candidates:

        print("\n----------------------------------------")

        print(
            f"Rank: #{candidate['rank']}"
        )

        print(
            f"Resume: {candidate['resume_file']}"
        )

        print(
            f"Candidate: {candidate['name']}"
        )

        print(
            f"Score: {candidate['score']}/100"
        )

        print(
            f"Recommendation: "
            f"{candidate['recommendation']}"
        )

        print(
            f"Experience: "
            f"{candidate['total_experience_years']} years"
        )


        print("\nSummary:")

        print(
            candidate["summary"]
        )


        print("\nMatched Required Skills:")

        for skill in candidate[
            "matched_required_skills"
        ]:

            print(
                f"  ✓ {skill}"
            )


        print("\nMissing Required Skills:")

        for skill in candidate[
            "missing_required_skills"
        ]:

            print(
                f"  ✗ {skill}"
            )


        print("\nStrengths:")

        for strength in candidate[
            "strengths"
        ]:

            print(
                f"  + {strength}"
            )


        print("\nWeaknesses:")

        for weakness in candidate[
            "weaknesses"
        ]:

            print(
                f"  - {weakness}"
            )


        print("\nWhy Good Fit:")

        for reason in candidate[
            "why_good_fit"
        ]:

            print(
                f"  → {reason}"
            )


        print("\nConcerns:")

        for concern in candidate[
            "concerns"
        ]:

            print(
                f"  ! {concern}"
            )


    # --------------------------------------------------------
    # STEP 6
    # --------------------------------------------------------

    print("\n========================================")
    print("SAVING RESULTS")
    print("========================================")


    with open(
        "results.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            candidates,
            file,
            indent=2,
            ensure_ascii=False
        )


    print(
        "Results saved to results.json"
    )


# ============================================================
# 14. RUN PROGRAM
# ============================================================

if __name__ == "__main__":

    main()
