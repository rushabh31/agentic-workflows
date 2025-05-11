"""
True Multi-Agent SOAP Note Evaluation Framework (MASEF)

This notebook implements a genuinely interactive multi-agent system for generating
and evaluating SOAP notes from healthcare transcripts, with true agent-to-agent communication.
"""

# ======================================================================
# PART 0: ENVIRONMENT SETUP AND DEPENDENCIES
# ======================================================================

# Google Colab environment setup
!pip install -q langchain langchain_groq langgraph transformers openai matplotlib seaborn pandas numpy plotly networkx ipywidgets
  
import os
import json
import time
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Any, Tuple, Optional, Union, Set, Callable
from datetime import datetime
from enum import Enum
import uuid

# Setup the Groq API key (needs to be provided by the user)
os.environ["GROQ_API_KEY"] = "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"  # Replace with your actual Groq API key

# Langchain imports
from langchain.schema import Document
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain.schema.messages import SystemMessage, HumanMessage, AIMessage
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.output_parser import StrOutputParser

# Langgraph imports for agent orchestration
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

# Pydantic for data modeling
from pydantic import BaseModel, Field, validator

# ======================================================================
# PART 1: DATA MODELS
# ======================================================================
class DispositionType(str, Enum):
    """Enum for different call disposition types."""
    AUTHORIZATION = "authorization"
    CLAIMS_INQUIRY = "claims_inquiry"
    BENEFITS_EXPLANATION = "benefits_explanation"
    GRIEVANCE = "grievance"
    ENROLLMENT = "enrollment"
    OTHER = "other"

class SentimentType(str, Enum):
    """Enum for sentiment analysis types."""
    SATISFIED = "satisfied"
    NEUTRAL = "neutral"
    DISSATISFIED = "dissatisfied"

class SOAPNote(BaseModel):
    """Model representing a complete SOAP note."""
    subjective: str
    objective: str
    assessment: str
    plan: str
    
    def __str__(self):
        return f"SUBJECTIVE:\n{self.subjective}\n\nOBJECTIVE:\n{self.objective}\n\nASSESSMENT:\n{self.assessment}\n\nPLAN:\n{self.plan}"
    
    def to_dict(self):
        return {
            "subjective": self.subjective,
            "objective": self.objective,
            "assessment": self.assessment,
            "plan": self.plan
        }

class Finding(BaseModel):
    """A specific finding from an agent."""
    content: str
    confidence: float = Field(ge=0.0, le=1.0, default=0.8)
    source: str
    category: str
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

class Question(BaseModel):
    """A question from one agent to another."""
    content: str
    from_agent: str
    to_agent: str
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    resolved: bool = False
    resolution: Optional[str] = None

class Message(BaseModel):
    """A message in the agent conversation."""
    sender: str
    content: str
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    message_type: str = "statement"  # Can be "statement", "question", "response", "finding"
    reference_id: Optional[str] = None  # For responses to specific messages
    
class AgentState(BaseModel):
    """Complete state for the multi-agent system with conversation."""
    transcript: str
    messages: List[Message] = Field(default_factory=list)
    questions: List[Question] = Field(default_factory=list)
    findings: List[Finding] = Field(default_factory=list)
    disposition: Optional[DispositionType] = None
    conversation_rounds: int = 0
    max_rounds: int = 3  # Prevent infinite loops
    final_soap: Optional[SOAPNote] = None

class EvaluationDimension(str, Enum):
    """Enum for evaluation dimensions."""
    ACCURACY = "accuracy"
    COMPLETENESS = "completeness"
    RELEVANCE = "relevance"
    STRUCTURE = "structure"
    COHERENCE = "coherence"

class DimensionScore(BaseModel):
    """Score for a specific evaluation dimension."""
    dimension: EvaluationDimension
    score: float = Field(ge=0.0, le=10.0)
    confidence: float = Field(ge=0.0, le=1.0)
    justification: str
    
class EvaluationResult(BaseModel):
    """Overall evaluation result for a SOAP note."""
    accuracy: float = Field(ge=0.0, le=10.0)
    completeness: float = Field(ge=0.0, le=10.0)
    relevance: float = Field(ge=0.0, le=10.0)
    structure: float = Field(ge=0.0, le=10.0)
    coherence: float = Field(ge=0.0, le=10.0)
    overallQuality: float = Field(ge=0.0, le=10.0)
    dimensionScores: List[DimensionScore]
    
class ComparativeEvaluation(BaseModel):
    """Comparative evaluation between sequential and multi-agent approaches."""
    sequentialScore: EvaluationResult
    multiAgentScore: EvaluationResult
    comparison: str
    improvementPercentage: float
    errorPatterns: Dict[str, List[str]]
    recommendations: List[str]

# ======================================================================
# PART 2: GROQ LLM CONFIGURATION
# ======================================================================

def get_llm(model_name="gemma2-9b-it", temperature=0.2):
    """Create and return a Groq LLM instance."""
    return ChatGroq(
        model_name=model_name,
        temperature=temperature,
        max_tokens=4096
    )

# Default LLM for general use
llm = get_llm()

# ======================================================================
# PART 3: SEQUENTIAL PIPELINE IMPLEMENTATION
# ======================================================================

def extract_soap_sections(text):
    """
    Extract SOAP sections from generated text.
    
    Args:
        text: The text containing SOAP sections
        
    Returns:
        dict: The extracted sections
    """
    sections = {}
    
    # Try to extract each section using regex
    subjective_match = re.search(r'(?:Subjective:|SUBJECTIVE:)(.*?)(?:Objective:|OBJECTIVE:|$)', text, re.DOTALL)
    if subjective_match:
        sections["subjective"] = subjective_match.group(1).strip()
    
    objective_match = re.search(r'(?:Objective:|OBJECTIVE:)(.*?)(?:Assessment:|ASSESSMENT:|$)', text, re.DOTALL)
    if objective_match:
        sections["objective"] = objective_match.group(1).strip()
    
    assessment_match = re.search(r'(?:Assessment:|ASSESSMENT:)(.*?)(?:Plan:|PLAN:|$)', text, re.DOTALL)
    if assessment_match:
        sections["assessment"] = assessment_match.group(1).strip()
    
    plan_match = re.search(r'(?:Plan:|PLAN:)(.*?)$', text, re.DOTALL)
    if plan_match:
        sections["plan"] = plan_match.group(1).strip()
    
    return sections

def identify_disposition(transcript: str) -> DispositionType:
    """
    Step 1: Identify the call disposition.
    
    Args:
        transcript: The call transcript text
        
    Returns:
        DispositionType: The identified disposition
    """
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            "You are a healthcare contact center expert. Your task is to identify the disposition (type) of this call. "
            "The possible dispositions are: authorization, claims_inquiry, benefits_explanation, grievance, enrollment, other. "
            "Return only the disposition category name without any additional explanation."
        ),
        HumanMessagePromptTemplate.from_template("Transcript: {transcript}\n\nWhat is the disposition of this call?")
    ])

    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({"transcript": transcript})
    
    # Normalize the disposition
    normalized_result = result.lower().strip()
    disposition = DispositionType.OTHER
    
    # Map the result to a disposition type
    for disp in DispositionType:
        if disp.value in normalized_result:
            disposition = disp
            break
    
    return disposition

def generate_soap_note(transcript: str, disposition: DispositionType) -> SOAPNote:
    """
    Generate a SOAP note with disposition-specific prompting.
    
    Args:
        transcript: The call transcript text
        disposition: The identified call disposition
        
    Returns:
        SOAPNote: The generated SOAP note
    """
    # Different prompt templates based on disposition type
    disposition_prompts = {
        DispositionType.AUTHORIZATION: 
            "For this authorization call, focus on: procedure details, medical necessity, provider information, timeline, and authorization status.",
        DispositionType.CLAIMS_INQUIRY: 
            "For this claims inquiry call, focus on: claim details, dates of service, denial reasons, amounts, and appeal opportunities.",
        DispositionType.BENEFITS_EXPLANATION: 
            "For this benefits explanation call, focus on: benefit inquiries, coverage limitations, cost-sharing, and authorization requirements.",
        DispositionType.GRIEVANCE: 
            "For this grievance call, focus on: the specific complaint, timeline of events, previous resolution attempts, and requested resolution.",
        DispositionType.ENROLLMENT: 
            "For this enrollment call, focus on: enrollment status/request details, eligibility factors, plan selection, and effective dates.",
        DispositionType.OTHER: 
            "Extract key information from this healthcare call, including member concerns, factual details, situation assessment, and action items."
    }
    
    prompt_template = disposition_prompts.get(disposition, disposition_prompts[DispositionType.OTHER])
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            "You are a clinical documentation specialist in a healthcare contact center. "
            "Create a comprehensive SOAP note (Subjective, Objective, Assessment, Plan) based on the call transcript. "
            f"{prompt_template} "
            "Format your response with clear sections for Subjective, Objective, Assessment, and Plan."
        ),
        HumanMessagePromptTemplate.from_template("Transcript: {transcript}\n\nGenerate a SOAP note:")
    ])

    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({"transcript": transcript})
    
    # Parse the SOAP sections from the result
    section_texts = extract_soap_sections(result)
    
    return SOAPNote(
        subjective=section_texts.get("subjective", "Not available"),
        objective=section_texts.get("objective", "Not available"),
        assessment=section_texts.get("assessment", "Not available"),
        plan=section_texts.get("plan", "Not available")
    )

def run_sequential_pipeline(transcript: str) -> SOAPNote:
    """
    Run the complete sequential pipeline on a transcript.
    
    Args:
        transcript: The call transcript text
        
    Returns:
        SOAPNote: The generated SOAP note
    """
    # Step 1: Identify disposition
    disposition = identify_disposition(transcript)
    print(f"Identified disposition: {disposition}")
    
    # Step 2: Generate SOAP note
    soap_note = generate_soap_note(transcript, disposition)
    print("Generated SOAP note using sequential pipeline")
    
    return soap_note

# ======================================================================
# PART 4: TRUE MULTI-AGENT SYSTEM WITH INTERACTIVE COMMUNICATION
# ======================================================================

# Define agent roles and specialties
class AgentRole(str, Enum):
    """Enum for agent roles in the multi-agent system."""
    COORDINATOR = "coordinator"
    MEDICAL_SPECIALIST = "medical_specialist"
    DOCUMENTATION_SPECIALIST = "documentation_specialist"
    BILLING_SPECIALIST = "billing_specialist"
    TRIAGE_SPECIALIST = "triage_specialist"
    SOAP_GENERATOR = "soap_generator"

def format_messages_for_prompt(messages: List[Message]) -> str:
    """Format the message history for inclusion in prompts."""
    formatted = []
    for msg in messages:
        formatted.append(f"[{msg.sender}]: {msg.content}")
    return "\n".join(formatted)

def triage_agent(state: AgentState) -> AgentState:
    """
    Initial triage agent that identifies call type and starts the team conversation.
    """
    transcript = state.transcript
    
    # Identify disposition
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            "You are a triage specialist in a healthcare contact center. "
            "Your job is to quickly analyze calls and route them to the appropriate specialists. "
            "Identify the call type (authorization, claims_inquiry, benefits_explanation, grievance, "
            "enrollment, or other) and briefly summarize the key issues for your team members."
        ),
        HumanMessagePromptTemplate.from_template("Transcript: {transcript}\n\nProvide your triage analysis:")
    ])
    
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({"transcript": transcript})
    
    # Determine disposition from the triage result
    disposition = DispositionType.OTHER
    for disp in DispositionType:
        if disp.value in result.lower():
            disposition = disp
            break
    
    # Create the initial message
    message = Message(
        sender=AgentRole.TRIAGE_SPECIALIST,
        content=result,
        message_type="statement"
    )
    
    # Update state
    new_state = state.copy()
    new_state.disposition = disposition
    new_state.messages.append(message)
    new_state.conversation_rounds += 1
    
    return new_state

def medical_specialist_agent(state: AgentState) -> AgentState:
    """
    Medical specialist agent that analyzes clinical aspects of the call.
    """
    transcript = state.transcript
    messages = state.messages
    formatted_messages = format_messages_for_prompt(messages)
    disposition = state.disposition
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            "You are a medical specialist in a multi-agent healthcare team. "
            "Your role is to focus on clinical aspects: symptoms, diagnoses, medical necessity, "
            "treatment plans, and clinical documentation. Review the transcript and conversation, "
            "then provide your medical insights. Identify 2-3 key medical findings with confidence levels. "
            "If you need clarification from other specialists, ask specific questions. "
            f"This is a {disposition.value} call."
        ),
        HumanMessagePromptTemplate.from_template(
            "Transcript: {transcript}\n\n"
            "Team conversation so far:\n{messages}\n\n"
            "Provide your medical specialist analysis:"
        )
    ])
    
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({
        "transcript": transcript,
        "messages": formatted_messages
    })
    
    # Create the response message
    message = Message(
        sender=AgentRole.MEDICAL_SPECIALIST,
        content=result,
        message_type="statement"
    )
    
    # Extract findings and questions
    findings = []
    questions = []
    
    # Parse for findings (looking for statements with confidence levels)
    finding_matches = re.finditer(r'(finding|observation):\s*(.+?)(?:\s*\(confidence:?\s*([0-9.]+)\))?', result.lower())
    for match in finding_matches:
        content = match.group(2).strip()
        confidence_str = match.group(3)
        confidence = float(confidence_str) if confidence_str else 0.8
        
        findings.append(Finding(
            content=content,
            confidence=min(1.0, max(0.1, confidence)),
            source=AgentRole.MEDICAL_SPECIALIST,
            category="medical"
        ))
    
    # Parse for questions (looking for direct questions to other agents)
    question_matches = re.finditer(r'(?:question for|@)(?:\s*)(billing|documentation|coordinator)(?:\s*):?\s*(.+?)(?:\?|$)', result.lower())
    for match in question_matches:
        to_agent = match.group(1).strip()
        content = match.group(2).strip() + "?"
        
        # Map to formal agent role
        to_role = AgentRole.COORDINATOR
        if "billing" in to_agent:
            to_role = AgentRole.BILLING_SPECIALIST
        elif "documentation" in to_agent:
            to_role = AgentRole.DOCUMENTATION_SPECIALIST
        
        questions.append(Question(
            content=content,
            from_agent=AgentRole.MEDICAL_SPECIALIST,
            to_agent=to_role
        ))
    
    # Update state
    new_state = state.copy()
    new_state.messages.append(message)
    new_state.findings.extend(findings)
    new_state.questions.extend(questions)
    
    return new_state

def billing_specialist_agent(state: AgentState) -> AgentState:
    """
    Billing specialist agent that analyzes insurance, claims, and financial aspects.
    """
    transcript = state.transcript
    messages = state.messages
    formatted_messages = format_messages_for_prompt(messages)
    disposition = state.disposition
    
    # Check if there are any questions directed to the billing specialist
    billing_questions = [q for q in state.questions if q.to_agent == AgentRole.BILLING_SPECIALIST and not q.resolved]
    questions_text = "\n".join([f"Question from {q.from_agent}: {q.content}" for q in billing_questions])
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            "You are a billing/insurance specialist in a multi-agent healthcare team. "
            "Your role is to focus on insurance coverage, claims processing, authorizations, "
            "benefits, cost-sharing, and financial aspects. Review the transcript and conversation, "
            "then provide your billing/insurance insights. Identify 2-3 key billing findings with confidence levels. "
            "If you need clarification from other specialists, ask specific questions. "
            f"This is a {disposition.value} call."
            + (f"\n\nPlease address these questions from your colleagues:\n{questions_text}" if questions_text else "")
        ),
        HumanMessagePromptTemplate.from_template(
            "Transcript: {transcript}\n\n"
            "Team conversation so far:\n{messages}\n\n"
            "Provide your billing/insurance specialist analysis:"
        )
    ])
    
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({
        "transcript": transcript,
        "messages": formatted_messages
    })
    
    # Create the response message
    message = Message(
        sender=AgentRole.BILLING_SPECIALIST,
        content=result,
        message_type="statement"
    )
    
    # Extract findings and questions
    findings = []
    questions = []
    
    # Parse for findings
    finding_matches = re.finditer(r'(finding|observation):\s*(.+?)(?:\s*\(confidence:?\s*([0-9.]+)\))?', result.lower())
    for match in finding_matches:
        content = match.group(2).strip()
        confidence_str = match.group(3)
        confidence = float(confidence_str) if confidence_str else 0.8
        
        findings.append(Finding(
            content=content,
            confidence=min(1.0, max(0.1, confidence)),
            source=AgentRole.BILLING_SPECIALIST,
            category="billing"
        ))
    
    # Parse for questions
    question_matches = re.finditer(r'(?:question for|@)(?:\s*)(medical|documentation|coordinator)(?:\s*):?\s*(.+?)(?:\?|$)', result.lower())
    for match in question_matches:
        to_agent = match.group(1).strip()
        content = match.group(2).strip() + "?"
        
        # Map to formal agent role
        to_role = AgentRole.COORDINATOR
        if "medical" in to_agent:
            to_role = AgentRole.MEDICAL_SPECIALIST
        elif "documentation" in to_agent:
            to_role = AgentRole.DOCUMENTATION_SPECIALIST
        
        questions.append(Question(
            content=content,
            from_agent=AgentRole.BILLING_SPECIALIST,
            to_agent=to_role
        ))
    
    # Mark questions as resolved
    resolved_questions = []
    for q in billing_questions:
        q.resolved = True
        q.resolution = result
        resolved_questions.append(q)
    
    # Update state
    new_state = state.copy()
    new_state.messages.append(message)
    new_state.findings.extend(findings)
    new_state.questions.extend(questions)
    
    # Update resolved questions
    for i, q in enumerate(new_state.questions):
        for rq in resolved_questions:
            if q.content == rq.content and q.from_agent == rq.from_agent and q.to_agent == rq.to_agent:
                new_state.questions[i] = rq
    
    return new_state

def documentation_specialist_agent(state: AgentState) -> AgentState:
    """
    Documentation specialist agent that focuses on healthcare documentation standards.
    """
    transcript = state.transcript
    messages = state.messages
    formatted_messages = format_messages_for_prompt(messages)
    disposition = state.disposition
    
    # Check if there are any questions directed to the documentation specialist
    doc_questions = [q for q in state.questions if q.to_agent == AgentRole.DOCUMENTATION_SPECIALIST and not q.resolved]
    questions_text = "\n".join([f"Question from {q.from_agent}: {q.content}" for q in doc_questions])
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            "You are a documentation specialist in a multi-agent healthcare team. "
            "Your role is to focus on proper documentation structure, compliance requirements, "
            "required elements for the SOAP note, and documentation best practices. "
            "Review the transcript and conversation, then provide your documentation insights. "
            "Identify 2-3 key documentation findings with confidence levels. "
            "If you need clarification from other specialists, ask specific questions. "
            f"This is a {disposition.value} call."
            + (f"\n\nPlease address these questions from your colleagues:\n{questions_text}" if questions_text else "")
        ),
        HumanMessagePromptTemplate.from_template(
            "Transcript: {transcript}\n\n"
            "Team conversation so far:\n{messages}\n\n"
            "Provide your documentation specialist analysis:"
        )
    ])
    
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({
        "transcript": transcript,
        "messages": formatted_messages
    })
    
    # Create the response message
    message = Message(
        sender=AgentRole.DOCUMENTATION_SPECIALIST,
        content=result,
        message_type="statement"
    )
    
    # Extract findings and questions
    findings = []
    questions = []
    
    # Parse for findings
    finding_matches = re.finditer(r'(finding|observation):\s*(.+?)(?:\s*\(confidence:?\s*([0-9.]+)\))?', result.lower())
    for match in finding_matches:
        content = match.group(2).strip()
        confidence_str = match.group(3)
        confidence = float(confidence_str) if confidence_str else 0.8
        
        findings.append(Finding(
            content=content,
            confidence=min(1.0, max(0.1, confidence)),
            source=AgentRole.DOCUMENTATION_SPECIALIST,
            category="documentation"
        ))
    
    # Parse for questions
    question_matches = re.finditer(r'(?:question for|@)(?:\s*)(medical|billing|coordinator)(?:\s*):?\s*(.+?)(?:\?|$)', result.lower())
    for match in question_matches:
        to_agent = match.group(1).strip()
        content = match.group(2).strip() + "?"
        
        # Map to formal agent role
        to_role = AgentRole.COORDINATOR
        if "medical" in to_agent:
            to_role = AgentRole.MEDICAL_SPECIALIST
        elif "billing" in to_agent:
            to_role = AgentRole.BILLING_SPECIALIST
        
        questions.append(Question(
            content=content,
            from_agent=AgentRole.DOCUMENTATION_SPECIALIST,
            to_agent=to_role
        ))
    
    # Mark questions as resolved
    resolved_questions = []
    for q in doc_questions:
        q.resolved = True
        q.resolution = result
        resolved_questions.append(q)
    
    # Update state
    new_state = state.copy()
    new_state.messages.append(message)
    new_state.findings.extend(findings)
    new_state.questions.extend(questions)
    
    # Update resolved questions
    for i, q in enumerate(new_state.questions):
        for rq in resolved_questions:
            if q.content == rq.content and q.from_agent == rq.from_agent and q.to_agent == rq.to_agent:
                new_state.questions[i] = rq
    
    return new_state

def coordinator_agent(state: AgentState) -> Dict[str, Any]:
    """
    Coordinator agent that manages the conversation and determines next steps.
    """
    transcript = state.transcript
    messages = state.messages
    formatted_messages = format_messages_for_prompt(messages)
    disposition = state.disposition
    findings = state.findings
    questions = state.questions
    
    # Format findings for the prompt
    formatted_findings = "\n".join([
        f"- {f.source} ({f.category}): {f.content} (confidence: {f.confidence:.1f})" 
        for f in findings
    ])
    
    # Check for unresolved questions
    unresolved_questions = [q for q in questions if not q.resolved]
    has_unresolved = len(unresolved_questions) > 0
    formatted_questions = "\n".join([
        f"- From {q.from_agent} to {q.to_agent}: {q.content}" 
        for q in unresolved_questions
    ])
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            "You are the coordinator in a multi-agent healthcare team. "
            "Your role is to manage the team conversation, identify information gaps, "
            "resolve conflicts, and determine when the team has sufficient information "
            "to generate the final SOAP note. "
            f"This is a {disposition.value} call. "
            "Based on the conversation so far, team findings, and unresolved questions, "
            "decide whether to: 1) have another round of discussion to resolve questions "
            "or gather more information, or 2) proceed to SOAP note generation.\n\n"
            f"Team findings so far:\n{formatted_findings}\n\n"
            + (f"Unresolved questions:\n{formatted_questions}\n\n" if has_unresolved else "No unresolved questions.\n\n")
        ),
        HumanMessagePromptTemplate.from_template(
            "Transcript: {transcript}\n\n"
            "Team conversation so far:\n{messages}\n\n"
            "Provide your coordinator assessment and decision:"
        )
    ])
    
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({
        "transcript": transcript,
        "messages": formatted_messages
    })
    
    # Create the response message
    message = Message(
        sender=AgentRole.COORDINATOR,
        content=result,
        message_type="statement"
    )
    
    # Determine if we need another round or can proceed to SOAP generation
    needs_another_round = (
        ("need more information" in result.lower() or 
         "another round" in result.lower() or 
         "more discussion" in result.lower() or
         "clarification needed" in result.lower()) and
        state.conversation_rounds < state.max_rounds and
        has_unresolved
    )
    
    # Update state
    new_state = state.copy()
    new_state.messages.append(message)
    new_state.conversation_rounds += 1
    
    # Return state with decision
    if needs_another_round:
        return {"next": "continue_discussion", "state": new_state}
    else:
        return {"next": "generate_soap", "state": new_state}

def soap_generator_agent(state: AgentState) -> AgentState:
    """
    SOAP generator agent that synthesizes all findings into a final SOAP note.
    """
    transcript = state.transcript
    messages = state.messages
    formatted_messages = format_messages_for_prompt(messages)
    disposition = state.disposition
    findings = state.findings
    
    # Format findings by category
    medical_findings = [f for f in findings if f.category == "medical"]
    billing_findings = [f for f in findings if f.category == "billing"]
    documentation_findings = [f for f in findings if f.category == "documentation"]
    
    formatted_medical = "\n".join([f"- {f.content} (confidence: {f.confidence:.1f})" for f in medical_findings])
    formatted_billing = "\n".join([f"- {f.content} (confidence: {f.confidence:.1f})" for f in billing_findings])
    formatted_documentation = "\n".join([f"- {f.content} (confidence: {f.confidence:.1f})" for f in documentation_findings])
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            "You are the SOAP note generator for a multi-agent healthcare team. "
            "Your role is to synthesize all the information gathered by the team into "
            "a comprehensive, well-structured SOAP note. "
            f"This is a {disposition.value} call. "
            "Use the transcript, team conversation, and specialist findings to create "
            "a complete SOAP note with Subjective, Objective, Assessment, and Plan sections. "
            "Follow proper healthcare documentation standards and ensure all relevant "
            "information is appropriately categorized.\n\n"
            f"Medical findings:\n{formatted_medical or 'None provided'}\n\n"
            f"Billing/Insurance findings:\n{formatted_billing or 'None provided'}\n\n"
            f"Documentation findings:\n{formatted_documentation or 'None provided'}\n\n"
        ),
        HumanMessagePromptTemplate.from_template(
            "Transcript: {transcript}\n\n"
            "Team conversation summary:\n{messages}\n\n"
            "Generate the complete SOAP note:"
        )
    ])
    
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({
        "transcript": transcript,
        "messages": formatted_messages
    })
    
    # Parse the SOAP sections
    section_texts = extract_soap_sections(result)
    
    soap_note = SOAPNote(
        subjective=section_texts.get("subjective", "Not available"),
        objective=section_texts.get("objective", "Not available"),
        assessment=section_texts.get("assessment", "Not available"),
        plan=section_texts.get("plan", "Not available")
    )
    
    # Create the final message
    message = Message(
        sender=AgentRole.SOAP_GENERATOR,
        content=f"Final SOAP Note Generated:\n\n{soap_note}",
        message_type="statement"
    )
    
    # Update state
    new_state = state.copy()
    new_state.messages.append(message)
    new_state.final_soap = soap_note
    
    return new_state

def create_true_multi_agent_workflow():
    """Create a true multi-agent system with interactive communication."""
    # Create the state graph using the state_schema parameter
    workflow = StateGraph(state_schema=AgentState)
    
    # Add nodes for each agent
    workflow.add_node("triage", triage_agent)
    workflow.add_node("medical_specialist", medical_specialist_agent)
    workflow.add_node("billing_specialist", billing_specialist_agent)
    workflow.add_node("documentation_specialist", documentation_specialist_agent)
    workflow.add_node("coordinator", coordinator_agent)
    workflow.add_node("soap_generator", soap_generator_agent)
    
    # Set the entry point
    workflow.set_entry_point("triage")
    
    # Initial specialist consultation
    workflow.add_edge("triage", "medical_specialist")
    workflow.add_edge("medical_specialist", "billing_specialist")
    workflow.add_edge("billing_specialist", "documentation_specialist")
    workflow.add_edge("documentation_specialist", "coordinator")
    
    # Decision point - continue discussion or generate SOAP
    workflow.add_conditional_edges(
        "coordinator",
        lambda x: x["next"],
        {
            "continue_discussion": "medical_specialist",  # Loop back for more discussion
            "generate_soap": "soap_generator"            # Proceed to SOAP generation
        }
    )
    
    # Final output
    workflow.add_edge("soap_generator", END)
    
    return workflow.compile()


def run_multi_agent_system(transcript: str) -> SOAPNote:
    """
    Run the multi-agent system on a transcript.
    
    Args:
        transcript: The call transcript text
        
    Returns:
        SOAPNote: The generated SOAP note
    """
    # Create the multi-agent workflow
    workflow = create_true_multi_agent_workflow()
    
    # Initialize the state with the transcript
    initial_state = AgentState(transcript=transcript)
    
    # Execute the workflow
    final_state = workflow.invoke(initial_state)
    
    # Print conversation for transparency
    print("\nAgent Conversation:")
    for i, msg in enumerate(final_state.messages):
        print(f"{i+1}. [{msg.sender}]: {msg.content[:150]}..." if len(msg.content) > 150 else f"{i+1}. [{msg.sender}]: {msg.content}")
    
    # Print findings
    print("\nAgent Findings:")
    findings_by_category = {}
    for finding in final_state.findings:
        if finding.category not in findings_by_category:
            findings_by_category[finding.category] = []
        findings_by_category[finding.category].append(finding)
    
    for category, findings in findings_by_category.items():
        print(f"\n{category.title()} Findings:")
        for i, finding in enumerate(findings):
            print(f"{i+1}. {finding.content} (confidence: {finding.confidence:.1f})")
    
    # Extract and return the SOAP note from the final state
    soap_note = final_state.final_soap
    if not soap_note:
        # If no SOAP note was generated, create a default one
        soap_note = SOAPNote(
            subjective="No subjective information was generated.",
            objective="No objective information was generated.",
            assessment="No assessment was generated.",
            plan="No plan was generated."
        )
    
    return soap_note

# ======================================================================
# PART 5: EVALUATION FRAMEWORK
# ======================================================================

# Define evaluator roles for specialized evaluation aspects
class EvaluatorRole(str, Enum):
    """Roles for specialized evaluator agents."""
    CLINICAL_ACCURACY = "clinical_accuracy_evaluator"
    DOCUMENTATION_COMPLETENESS = "documentation_completeness_evaluator"
    RELEVANCE = "relevance_evaluator"
    STRUCTURE = "structure_evaluator"
    COHERENCE = "coherence_evaluator"
    AP_LINKAGE = "assessment_plan_linkage_evaluator"
    META_EVALUATOR = "meta_evaluator"

# Specialized rubrics for each evaluator agent
EVALUATOR_RUBRICS = {
    EvaluatorRole.CLINICAL_ACCURACY: """
You are evaluating the clinical accuracy of a SOAP note. Focus on:
1. Factual Correctness (0-10): Are all statements in the SOAP note factually supported by the transcript?
2. Hallucination Detection (0-10): Does the note contain information not present in or implied by the transcript?
3. Medical Plausibility (0-10): Are the medical details and their relationships clinically plausible?
4. Omission of Critical Information (0-10): Has any critical clinical information from the transcript been omitted?

Guidelines for scoring:
- 9-10: Excellent - All information is accurate, no hallucinations, medically sound, no critical omissions
- 7-8: Good - Minor inaccuracies that don't affect clinical understanding, minimal non-critical omissions
- 5-6: Fair - Some inaccuracies or omissions that may slightly affect clinical understanding
- 3-4: Poor - Significant inaccuracies, some hallucinations, or important omissions
- 0-2: Very Poor - Major inaccuracies, substantial hallucinations, or critical omissions that could affect patient care

Provide specific examples of any errors, hallucinations, or omissions found.
""",

    EvaluatorRole.DOCUMENTATION_COMPLETENESS: """
You are evaluating the completeness of a SOAP note. Focus on:
1. SOAP Component Coverage (0-10): Are all four SOAP sections (Subjective, Objective, Assessment, Plan) present and populated?
2. Sub-Component Coverage (0-10): Are expected sub-components (e.g., HPI, PMH, ROS in Subjective) included where relevant?
3. Entity Recall (0-10): What proportion of key clinical entities from the transcript (symptoms, diagnoses, medications, procedures) are captured?
4. Detail Level (0-10): Is the level of detail appropriate for clinical understanding?

Guidelines for scoring:
- 9-10: Excellent - Comprehensive coverage of all components and relevant entities with appropriate detail
- 7-8: Good - All major components present with minor omissions in sub-components or entities
- 5-6: Fair - All main SOAP sections present but with notable gaps in sub-components or entities
- 3-4: Poor - Missing one or more SOAP sections or significant gaps in coverage
- 0-2: Very Poor - Multiple missing sections or critically incomplete

Provide specific examples of any missing components or entities that should have been included.
""",

    EvaluatorRole.RELEVANCE: """
You are evaluating the relevance of information in a SOAP note. Focus on:
1. Contextual Relevance (0-10): Is all information directly relevant to the patient's presenting problem or situation?
2. Noise Reduction (0-10): Has irrelevant or unnecessary information been excluded?
3. Clinical Significance (0-10): Is the information prioritized according to clinical importance?
4. Pertinence to Call Purpose (0-10): Does the note focus on addressing the primary reason for the call?

Guidelines for scoring:
- 9-10: Excellent - All information is highly relevant, no extraneous details, perfect prioritization
- 7-8: Good - Mostly relevant information with minor extraneous details
- 5-6: Fair - Some irrelevant information included or some prioritization issues
- 3-4: Poor - Significant inclusion of irrelevant details or poor prioritization
- 0-2: Very Poor - Dominated by irrelevant information or completely misses the call's purpose

Provide specific examples of irrelevant information included or relevant information that was deprioritized.
""",

    EvaluatorRole.STRUCTURE: """
You are evaluating the structural integrity of a SOAP note. Focus on:
1. SOAP Format Adherence (0-10): Are the four SOAP sections clearly delineated and in the correct order?
2. Information Categorization (0-10): Is information correctly placed in the appropriate SOAP section?
3. Internal Organization (0-10): Is information within each section logically organized (e.g., chronological HPI)?
4. Section Balance (0-10): Are the sections appropriately balanced in length and detail for the case?

Guidelines for scoring:
- 9-10: Excellent - Perfect adherence to SOAP format with ideal categorization and organization
- 7-8: Good - Clear SOAP structure with minor categorization or organization issues
- 5-6: Fair - Basic SOAP structure present but with noticeable categorization errors
- 3-4: Poor - Significant structural problems, major categorization errors
- 0-2: Very Poor - Fails to follow SOAP format or systematic miscategorization

Provide specific examples of structural issues, misplaced information, or organizational problems.
""",

    EvaluatorRole.COHERENCE: """
You are evaluating the coherence and readability of a SOAP note. Focus on:
1. Logical Flow (0-10): Does the note progress logically from one idea to the next within and across sections?
2. Clarity (0-10): Is the language clear, precise, and unambiguous?
3. Conciseness (0-10): Is the note appropriately concise without sacrificing necessary detail?
4. Terminology Consistency (0-10): Are medical terms used consistently throughout the note?

Guidelines for scoring:
- 9-10: Excellent - Seamless flow, crystal clear, optimally concise, perfectly consistent
- 7-8: Good - Generally good flow with minor clarity or conciseness issues
- 5-6: Fair - Readable but with noticeable flow, clarity, or consistency problems
- 3-4: Poor - Difficult to follow in places, unclear, or significantly inconsistent
- 0-2: Very Poor - Incoherent, confusing, verbose, or critically inconsistent

Provide specific examples of flow problems, unclear language, verbosity, or inconsistencies.
""",

    EvaluatorRole.AP_LINKAGE: """
You are evaluating the logical linkage between the Assessment and Plan sections. Focus on:
1. Problem-Plan Alignment (0-10): Does each element in the Plan clearly address a problem identified in the Assessment?
2. Completeness of Coverage (0-10): Does the Plan address all issues raised in the Assessment?
3. Clinical Appropriateness (0-10): Are the planned actions clinically appropriate for the assessed problems?
4. Actionability (0-10): Are the planned actions specific, measurable, achievable, relevant, and time-bound (SMART)?

Guidelines for scoring:
- 9-10: Excellent - Perfect alignment, complete coverage, clinically optimal, and highly actionable
- 7-8: Good - Strong alignment with minor gaps or slightly vague actions
- 5-6: Fair - Basic alignment but with noticeable gaps or somewhat vague actions
- 3-4: Poor - Significant misalignment, major gaps, or vague actions
- 0-2: Very Poor - Plan disconnected from Assessment or critically vague/inappropriate

Provide specific examples of misalignment, gaps in coverage, or vague/inappropriate plans.
""",

    EvaluatorRole.META_EVALUATOR: """
You are a meta-evaluator analyzing the assessments from multiple specialized evaluators. Your task is to:
1. Synthesize the findings from each specialized evaluator
2. Identify patterns, agreements, and disagreements among evaluators
3. Calculate weighted dimension scores and an overall quality score
4. Provide a final comparative analysis of the sequential vs. multi-agent SOAP notes

Consider these weights for the overall calculation:
- Clinical Accuracy: 30%
- Documentation Completeness: 25% 
- Coherence and Readability: 20%
- Structure: 15%
- Relevance: 10%

The formula for the overall quality score is:
Overall = 0.3×Accuracy + 0.25×Completeness + 0.2×Coherence + 0.15×Structure + 0.1×Relevance

Provide your final analysis with:
1. Summary of each evaluator's key findings
2. Calculated scores for each dimension (with confidence levels)
3. Overall quality scores for both approaches
4. Percentage improvement of one approach over the other
5. Detailed comparative analysis highlighting key strengths and weaknesses
"""
}

def create_evaluator_agent(role: EvaluatorRole, model_name="gemma2-9b-it", temperature=0):
    """
    Create a specialized evaluator agent with a specific role.
    
    Args:
        role: The evaluator role
        model_name: The model name to use
        temperature: The temperature parameter
        
    Returns:
        A function that evaluates a SOAP note
    """
    # Get the specialized rubric for this evaluator
    rubric = EVALUATOR_RUBRICS.get(role, "")
    
    # Create the LLM with appropriate settings
    evaluation_llm = get_llm(model_name=model_name, temperature=temperature)
    
    def evaluate_soap_note(transcript: str, sequential_soap: SOAPNote, multi_agent_soap: SOAPNote):
        """
        Evaluate both SOAP notes using this specialized evaluator.
        
        Args:
            transcript: The call transcript
            sequential_soap: The SOAP note from sequential pipeline
            multi_agent_soap: The SOAP note from multi-agent system
            
        Returns:
            Dict with evaluation results
        """
        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                f"You are a specialized {role.value.replace('_', ' ')}. "
                f"Your task is to evaluate and compare two SOAP notes generated for the same transcript.\n\n"
                f"{rubric}\n\n"
                f"Provide scores on a scale of 0-10 for each criterion, along with detailed justifications. "
                f"Then provide an overall score for your specialty area, and a confidence level "
                f"(Low, Medium, High) in your evaluation."
            ),
            HumanMessagePromptTemplate.from_template(
                "Transcript: {transcript}\n\n"
                "Sequential Pipeline SOAP Note:\n"
                "Subjective: {sequential_subjective}\n"
                "Objective: {sequential_objective}\n"
                "Assessment: {sequential_assessment}\n"
                "Plan: {sequential_plan}\n\n"
                "Multi-Agent System SOAP Note:\n"
                "Subjective: {multi_agent_subjective}\n"
                "Objective: {multi_agent_objective}\n"
                "Assessment: {multi_agent_assessment}\n"
                "Plan: {multi_agent_plan}\n\n"
                "Evaluate both SOAP notes for {role}:"
            )
        ])

        chain = prompt | evaluation_llm | StrOutputParser()
        result = chain.invoke({
            "transcript": transcript,
            "sequential_subjective": sequential_soap.subjective,
            "sequential_objective": sequential_soap.objective,
            "sequential_assessment": sequential_soap.assessment,
            "sequential_plan": sequential_soap.plan,
            "multi_agent_subjective": multi_agent_soap.subjective,
            "multi_agent_objective": multi_agent_soap.objective,
            "multi_agent_assessment": multi_agent_soap.assessment,
            "multi_agent_plan": multi_agent_soap.plan,
            "role": role.value.replace("_", " ")
        })
        
        # Parse the evaluation result to extract scores
        sequential_score, multi_agent_score, confidence = parse_evaluation_result(result)
        
        return {
            "role": role,
            "sequential_score": sequential_score,
            "multi_agent_score": multi_agent_score,
            "confidence": confidence,
            "justification": result
        }
    
    return evaluate_soap_note

def parse_evaluation_result(evaluation_text: str):
    """
    Parse the evaluation text from an evaluator agent to extract scores.
    
    Args:
        evaluation_text: The raw evaluation text
        
    Returns:
        Tuple of (sequential_score, multi_agent_score, confidence)
    """
    # Default values
    sequential_score = 5.0
    multi_agent_score = 5.0
    confidence = 0.5
    
    # Extract overall sequential score
    seq_match = re.search(r'(?:sequential|pipeline).*(?:overall|final)[^0-9]*([0-9](?:\.[0-9])?|10)(?:/| out of)10', evaluation_text.lower())
    if seq_match:
        sequential_score = float(seq_match.group(1))
    
    # Extract overall multi-agent score    
    ma_match = re.search(r'(?:multi-agent|multi agent).*(?:overall|final)[^0-9]*([0-9](?:\.[0-9])?|10)(?:/| out of)10', evaluation_text.lower())
    if ma_match:
        multi_agent_score = float(ma_match.group(1))
    
    # Extract confidence
    conf_match = re.search(r'confidence[^:]*:[^a-z]*(low|medium|high)', evaluation_text.lower())
    if conf_match:
        conf_level = conf_match.group(1).lower()
        if conf_level == "low":
            confidence = 0.3
        elif conf_level == "medium":
            confidence = 0.6
        elif conf_level == "high":
            confidence = 0.9
    
    return sequential_score, multi_agent_score, confidence

def create_meta_evaluator():
    """
    Create a meta-evaluator agent that synthesizes evaluations from specialized evaluators.
    
    Returns:
        A function that performs meta-evaluation
    """
    meta_llm = get_llm(model_name="gemma2-9b-it", temperature=0.1)
    
    def meta_evaluate(transcript: str, sequential_soap: SOAPNote, multi_agent_soap: SOAPNote, evaluations: List[Dict]):
        """
        Synthesize evaluations from specialized evaluators into a final comparative analysis.
        
        Args:
            transcript: The call transcript
            sequential_soap: The SOAP note from sequential pipeline
            multi_agent_soap: The SOAP note from multi-agent system
            evaluations: List of evaluations from specialized evaluators
            
        Returns:
            ComparativeEvaluation result
        """
        # Extract and format the evaluations for the prompt
        formatted_evals = ""
        for eval_result in evaluations:
            role = eval_result["role"].value.replace("_", " ").title()
            seq_score = eval_result["sequential_score"]
            ma_score = eval_result["multi_agent_score"]
            confidence = eval_result["confidence"]
            conf_level = "Low" if confidence < 0.4 else ("Medium" if confidence < 0.7 else "High")
            
            formatted_evals += f"## {role} Evaluation:\n"
            formatted_evals += f"- Sequential Pipeline Score: {seq_score}/10\n"
            formatted_evals += f"- Multi-Agent System Score: {ma_score}/10\n"
            formatted_evals += f"- Confidence Level: {conf_level}\n\n"
        
        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                f"You are a meta-evaluator synthesizing assessments from multiple specialized evaluators. "
                f"{EVALUATOR_RUBRICS[EvaluatorRole.META_EVALUATOR]}"
            ),
            HumanMessagePromptTemplate.from_template(
                "Transcript: {transcript}\n\n"
                "Sequential Pipeline SOAP Note:\n"
                "Subjective: {sequential_subjective}\n"
                "Objective: {sequential_objective}\n"
                "Assessment: {sequential_assessment}\n"
                "Plan: {sequential_plan}\n\n"
                "Multi-Agent System SOAP Note:\n"
                "Subjective: {multi_agent_subjective}\n"
                "Objective: {multi_agent_objective}\n"
                "Assessment: {multi_agent_assessment}\n"
                "Plan: {multi_agent_plan}\n\n"
                "Specialized Evaluator Results:\n{evaluations}\n\n"
                "Provide your meta-evaluation synthesizing these evaluations:"
            )
        ])

        chain = prompt | meta_llm | StrOutputParser()
        result = chain.invoke({
            "transcript": transcript,
            "sequential_subjective": sequential_soap.subjective,
            "sequential_objective": sequential_soap.objective,
            "sequential_assessment": sequential_soap.assessment,
            "sequential_plan": sequential_soap.plan,
            "multi_agent_subjective": multi_agent_soap.subjective,
            "multi_agent_objective": multi_agent_soap.objective,
            "multi_agent_assessment": multi_agent_soap.assessment,
            "multi_agent_plan": multi_agent_soap.plan,
            "evaluations": formatted_evals
        })
        
        # Extract information from the meta-evaluation
        improved_percentage = extract_improvement_percentage(result)
        error_patterns = extract_error_patterns(result)
        recommendations = extract_recommendations(result)
        
        # Calculate dimension scores based on evaluations
        accuracy_eval = next((e for e in evaluations if e["role"] == EvaluatorRole.CLINICAL_ACCURACY), None)
        completeness_eval = next((e for e in evaluations if e["role"] == EvaluatorRole.DOCUMENTATION_COMPLETENESS), None)
        relevance_eval = next((e for e in evaluations if e["role"] == EvaluatorRole.RELEVANCE), None)
        structure_eval = next((e for e in evaluations if e["role"] == EvaluatorRole.STRUCTURE), None)
        coherence_eval = next((e for e in evaluations if e["role"] == EvaluatorRole.COHERENCE), None)
        
        # Compile dimensional scores
        dimension_scores = []
        
        if accuracy_eval:
            dimension_scores.append(DimensionScore(
                dimension=EvaluationDimension.ACCURACY,
                score=accuracy_eval["sequential_score"],
                confidence=accuracy_eval["confidence"],
                justification="Based on clinical accuracy evaluation"
            ))
            
        if completeness_eval:
            dimension_scores.append(DimensionScore(
                dimension=EvaluationDimension.COMPLETENESS,
                score=completeness_eval["sequential_score"],
                confidence=completeness_eval["confidence"],
                justification="Based on documentation completeness evaluation"
            ))
            
        if relevance_eval:
            dimension_scores.append(DimensionScore(
                dimension=EvaluationDimension.RELEVANCE,
                score=relevance_eval["sequential_score"],
                confidence=relevance_eval["confidence"],
                justification="Based on relevance evaluation"
            ))
            
        if structure_eval:
            dimension_scores.append(DimensionScore(
                dimension=EvaluationDimension.STRUCTURE,
                score=structure_eval["sequential_score"],
                confidence=structure_eval["confidence"],
                justification="Based on structure evaluation"
            ))
            
        if coherence_eval:
            dimension_scores.append(DimensionScore(
                dimension=EvaluationDimension.COHERENCE,
                score=coherence_eval["sequential_score"],
                confidence=coherence_eval["confidence"],
                justification="Based on coherence evaluation"
            ))
        
        # Calculate overall quality scores
        sequential_overall = 0.0
        multi_agent_overall = 0.0
        
        if accuracy_eval:
            sequential_overall += 0.3 * accuracy_eval["sequential_score"]
            multi_agent_overall += 0.3 * accuracy_eval["multi_agent_score"]
            
        if completeness_eval:
            sequential_overall += 0.25 * completeness_eval["sequential_score"]
            multi_agent_overall += 0.25 * completeness_eval["multi_agent_score"]
            
        if coherence_eval:
            sequential_overall += 0.2 * coherence_eval["sequential_score"]
            multi_agent_overall += 0.2 * coherence_eval["multi_agent_score"]
            
        if structure_eval:
            sequential_overall += 0.15 * structure_eval["sequential_score"]
            multi_agent_overall += 0.15 * structure_eval["multi_agent_score"]
            
        if relevance_eval:
            sequential_overall += 0.1 * relevance_eval["sequential_score"]
            multi_agent_overall += 0.1 * relevance_eval["multi_agent_score"]
        
        # Create evaluation results
        sequential_eval = EvaluationResult(
            accuracy=accuracy_eval["sequential_score"] if accuracy_eval else 5.0,
            completeness=completeness_eval["sequential_score"] if completeness_eval else 5.0,
            relevance=relevance_eval["sequential_score"] if relevance_eval else 5.0,
            structure=structure_eval["sequential_score"] if structure_eval else 5.0,
            coherence=coherence_eval["sequential_score"] if coherence_eval else 5.0,
            overallQuality=sequential_overall,
            dimensionScores=dimension_scores
        )
        
        multi_agent_eval = EvaluationResult(
            accuracy=accuracy_eval["multi_agent_score"] if accuracy_eval else 5.0,
            completeness=completeness_eval["multi_agent_score"] if completeness_eval else 5.0,
            relevance=relevance_eval["multi_agent_score"] if relevance_eval else 5.0,
            structure=structure_eval["multi_agent_score"] if structure_eval else 5.0,
            coherence=coherence_eval["multi_agent_score"] if coherence_eval else 5.0,
            overallQuality=multi_agent_overall,
            dimensionScores=dimension_scores
        )
        
        return ComparativeEvaluation(
            sequentialScore=sequential_eval,
            multiAgentScore=multi_agent_eval,
            comparison=result,
            improvementPercentage=improved_percentage,
            errorPatterns=error_patterns,
            recommendations=recommendations
        )
    
    return meta_evaluate

def extract_improvement_percentage(text: str) -> float:
    """Extract the improvement percentage from meta-evaluation text."""
    improvement = 0.0
    
    # Look for percentage mentions
    matches = re.finditer(r'(?:improvement|increase|better by)[^0-9]*?([0-9]+(?:\.[0-9]+)?)(?:\s*)?%', text.lower())
    for match in matches:
        try:
            improvement = float(match.group(1))
            break
        except ValueError:
            continue
    
    return improvement

def extract_error_patterns(text: str) -> Dict[str, List[str]]:
    """Extract error patterns from meta-evaluation text."""
    error_patterns = {
        "sequential": ["None identified"],
        "multi_agent": ["None identified"]
    }
    
    # Look for sequential error patterns
    seq_section = re.search(r'(?:sequential|pipeline)(?:.*)(?:error patterns|weaknesses|limitations)[:;]?\s*((?:.+\n)+)', text.lower())
    if seq_section:
        seq_errors = re.findall(r'(?:[-•*]\s*|\d+\.\s*)(.+?)(?:\n|$)', seq_section.group(1))
        if seq_errors:
            error_patterns["sequential"] = [e.strip() for e in seq_errors if len(e.strip()) > 5]
    
    # Look for multi-agent error patterns
    ma_section = re.search(r'(?:multi-agent|multi agent)(?:.*)(?:error patterns|weaknesses|limitations)[:;]?\s*((?:.+\n)+)', text.lower())
    if ma_section:
        ma_errors = re.findall(r'(?:[-•*]\s*|\d+\.\s*)(.+?)(?:\n|$)', ma_section.group(1))
        if ma_errors:
            error_patterns["multi_agent"] = [e.strip() for e in ma_errors if len(e.strip()) > 5]
    
    return error_patterns

def extract_recommendations(text: str) -> List[str]:
    """Extract recommendations from meta-evaluation text."""
    recommendations = ["None provided"]
    
    # Look for recommendations section
    rec_section = re.search(r'(?:recommendations|suggestions|improvements|future work)[:;]?\s*((?:.+\n)+)', text.lower())
    if rec_section:
        recs = re.findall(r'(?:[-•*]\s*|\d+\.\s*)(.+?)(?:\n|$)', rec_section.group(1))
        if recs:
            recommendations = [r.strip() for r in recs if len(r.strip()) > 5]
    
    return recommendations

def evaluate_soap_notes(transcript: str, sequential_soap: SOAPNote, multi_agent_soap: SOAPNote) -> ComparativeEvaluation:
    """
    Evaluate SOAP notes using multiple specialized evaluator agents.
    
    Args:
        transcript: The call transcript
        sequential_soap: The SOAP note from sequential pipeline
        multi_agent_soap: The SOAP note from multi-agent system
        
    Returns:
        ComparativeEvaluation: The comparative evaluation results
    """
    # Create specialized evaluator agents
    evaluators = {
        EvaluatorRole.CLINICAL_ACCURACY: create_evaluator_agent(EvaluatorRole.CLINICAL_ACCURACY),
        EvaluatorRole.DOCUMENTATION_COMPLETENESS: create_evaluator_agent(EvaluatorRole.DOCUMENTATION_COMPLETENESS),
        EvaluatorRole.RELEVANCE: create_evaluator_agent(EvaluatorRole.RELEVANCE),
        EvaluatorRole.STRUCTURE: create_evaluator_agent(EvaluatorRole.STRUCTURE),
        EvaluatorRole.COHERENCE: create_evaluator_agent(EvaluatorRole.COHERENCE),
        EvaluatorRole.AP_LINKAGE: create_evaluator_agent(EvaluatorRole.AP_LINKAGE)
    }
    
    # Collect evaluations from each specialized agent
    evaluations = []
    for role, evaluator in evaluators.items():
        print(f"Running {role.value} evaluation...")
        evaluation = evaluator(transcript, sequential_soap, multi_agent_soap)
        evaluations.append(evaluation)
        print(f"  - Sequential: {evaluation['sequential_score']:.1f}/10")
        print(f"  - Multi-Agent: {evaluation['multi_agent_score']:.1f}/10")
    
    # Create and run meta-evaluator
    print("Running meta-evaluation...")
    meta_evaluator = create_meta_evaluator()
    comparative_eval = meta_evaluator(transcript, sequential_soap, multi_agent_soap, evaluations)
    
    return comparative_eval

# ======================================================================
# PART 6: VISUALIZATION AND REPORTING
# ======================================================================

def visualize_comparative_evaluation(eval_result: ComparativeEvaluation):
    """
    Create visualizations for comparative evaluation results.
    
    Args:
        eval_result: The comparative evaluation result
    """
    # Extract scores for visualization
    seq_scores = [
        eval_result.sequentialScore.accuracy,
        eval_result.sequentialScore.completeness,
        eval_result.sequentialScore.relevance,
        eval_result.sequentialScore.structure,
        eval_result.sequentialScore.coherence,
        eval_result.sequentialScore.overallQuality
    ]
    
    ma_scores = [
        eval_result.multiAgentScore.accuracy,
        eval_result.multiAgentScore.completeness,
        eval_result.multiAgentScore.relevance,
        eval_result.multiAgentScore.structure,
        eval_result.multiAgentScore.coherence,
        eval_result.multiAgentScore.overallQuality
    ]
    
    categories = ['Accuracy', 'Completeness', 'Relevance', 'Structure', 'Coherence', 'Overall']
    
    # Create a radar chart for the dimensions
    fig = plt.figure(figsize=(12, 6))
    
    # Bar chart
    ax1 = fig.add_subplot(121)
    x = np.arange(len(categories))
    width = 0.35
    
    ax1.bar(x - width/2, seq_scores, width, label='Sequential Pipeline')
    ax1.bar(x + width/2, ma_scores, width, label='Multi-Agent System')
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories)
    ax1.set_ylim(0, 10)
    ax1.set_ylabel('Score (0-10)')
    ax1.set_title('Comparative Scores by Dimension')
    ax1.legend()
    
    # Radar chart
    ax2 = fig.add_subplot(122, polar=True)
    
    # Number of variables
    N = len(categories) - 1  # Exclude "Overall" for the radar chart
    
    # What will be the angle of each axis in the plot
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Radar chart data
    seq_data = seq_scores[:-1]  # Exclude "Overall"
    seq_data += seq_data[:1]    # Close the loop
    
    ma_data = ma_scores[:-1]    # Exclude "Overall"
    ma_data += ma_data[:1]      # Close the loop
    
    # Plot radar chart
    ax2.plot(angles, seq_data, linewidth=1, linestyle='solid', label='Sequential Pipeline')
    ax2.fill(angles, seq_data, alpha=0.1)
    
    ax2.plot(angles, ma_data, linewidth=1, linestyle='solid', label='Multi-Agent System')
    ax2.fill(angles, ma_data, alpha=0.1)
    
    # Add labels
    plt.xticks(angles[:-1], categories[:-1])
    
    ax2.set_title('Dimensional Quality Profile')
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    plt.show()
    
    # Display improvement metrics
    improvement_percentages = [(m - s) / s * 100 if s > 0 else 0 for s, m in zip(seq_scores, ma_scores)]
    
    plt.figure(figsize=(10, 6))
    plt.bar(categories, improvement_percentages)
    plt.axhline(y=0, color='r', linestyle='-')
    plt.ylabel('Improvement (%)')
    plt.title('Multi-Agent System Improvement over Sequential Pipeline')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # Display error patterns
    print("\n=== Error Patterns Analysis ===")
    print("\nSequential Pipeline Errors:")
    for i, error in enumerate(eval_result.errorPatterns.get("sequential", [])):
        print(f"{i+1}. {error}")
    
    print("\nMulti-Agent System Errors:")
    for i, error in enumerate(eval_result.errorPatterns.get("multi_agent", [])):
        print(f"{i+1}. {error}")
    
    print("\n=== Recommendations ===")
    for i, rec in enumerate(eval_result.recommendations):
        print(f"{i+1}. {rec}")
    
    # Display overall comparison
    print(f"\nOverall Quality: Sequential Pipeline: {eval_result.sequentialScore.overallQuality:.1f}/10 vs Multi-Agent System: {eval_result.multiAgentScore.overallQuality:.1f}/10")
    print(f"Improvement: {eval_result.improvementPercentage:.1f}%")

def generate_text_report(transcript: str, sequential_soap: SOAPNote, multi_agent_soap: SOAPNote, eval_result: ComparativeEvaluation):
    """
    Generate a text-based report for the comparative evaluation.
    
    Args:
        transcript: The call transcript
        sequential_soap: The SOAP note from sequential pipeline
        multi_agent_soap: The SOAP note from multi-agent system
        eval_result: The comparative evaluation result
        
    Returns:
        str: Formatted text report
    """
    report = []
    
    # Title and introduction
    report.append("=" * 80)
    report.append("SOAP NOTE EVALUATION REPORT")
    report.append("=" * 80)
    report.append("")
    
    # Transcript summary (first 200 chars)
    report.append("TRANSCRIPT SUMMARY:")
    report.append("-" * 40)
    report.append(transcript[:200] + "..." if len(transcript) > 200 else transcript)
    report.append("")
    
    # Sequential SOAP note
    report.append("SEQUENTIAL PIPELINE SOAP NOTE:")
    report.append("-" * 40)
    report.append(f"SUBJECTIVE:\n{sequential_soap.subjective}")
    report.append(f"OBJECTIVE:\n{sequential_soap.objective}")
    report.append(f"ASSESSMENT:\n{sequential_soap.assessment}")
    report.append(f"PLAN:\n{sequential_soap.plan}")
    report.append("")
    
    # Multi-agent SOAP note
    report.append("MULTI-AGENT SYSTEM SOAP NOTE:")
    report.append("-" * 40)
    report.append(f"SUBJECTIVE:\n{multi_agent_soap.subjective}")
    report.append(f"OBJECTIVE:\n{multi_agent_soap.objective}")
    report.append(f"ASSESSMENT:\n{multi_agent_soap.assessment}")
    report.append(f"PLAN:\n{multi_agent_soap.plan}")
    report.append("")
    
    # Evaluation scores
    report.append("EVALUATION SCORES:")
    report.append("-" * 40)
    report.append(f"                     Sequential   Multi-Agent   Improvement")
    report.append(f"Accuracy:            {eval_result.sequentialScore.accuracy:.1f}/10       {eval_result.multiAgentScore.accuracy:.1f}/10        {(eval_result.multiAgentScore.accuracy - eval_result.sequentialScore.accuracy) / eval_result.sequentialScore.accuracy * 100:.1f}%")
    report.append(f"Completeness:        {eval_result.sequentialScore.completeness:.1f}/10       {eval_result.multiAgentScore.completeness:.1f}/10        {(eval_result.multiAgentScore.completeness - eval_result.sequentialScore.completeness) / eval_result.sequentialScore.completeness * 100:.1f}%")
    report.append(f"Relevance:           {eval_result.sequentialScore.relevance:.1f}/10       {eval_result.multiAgentScore.relevance:.1f}/10        {(eval_result.multiAgentScore.relevance - eval_result.sequentialScore.relevance) / eval_result.sequentialScore.relevance * 100:.1f}%")
    report.append(f"Structure:           {eval_result.sequentialScore.structure:.1f}/10       {eval_result.multiAgentScore.structure:.1f}/10        {(eval_result.multiAgentScore.structure - eval_result.sequentialScore.structure) / eval_result.sequentialScore.structure * 100:.1f}%")
    report.append(f"Coherence:           {eval_result.sequentialScore.coherence:.1f}/10       {eval_result.multiAgentScore.coherence:.1f}/10        {(eval_result.multiAgentScore.coherence - eval_result.sequentialScore.coherence) / eval_result.sequentialScore.coherence * 100:.1f}%")
    report.append(f"OVERALL QUALITY:     {eval_result.sequentialScore.overallQuality:.1f}/10       {eval_result.multiAgentScore.overallQuality:.1f}/10        {eval_result.improvementPercentage:.1f}%")
    report.append("")
    
    # Error patterns
    report.append("ERROR PATTERNS:")
    report.append("-" * 40)
    report.append("Sequential Pipeline:")
    for i, error in enumerate(eval_result.errorPatterns.get("sequential", [])):
        report.append(f"{i+1}. {error}")
    report.append("")
    
    report.append("Multi-Agent System:")
    for i, error in enumerate(eval_result.errorPatterns.get("multi_agent", [])):
        report.append(f"{i+1}. {error}")
    report.append("")
    
    # Recommendations
    report.append("RECOMMENDATIONS:")
    report.append("-" * 40)
    for i, rec in enumerate(eval_result.recommendations):
        report.append(f"{i+1}. {rec}")
    report.append("")
    
    # Join all parts
    return "\n".join(report)

# ======================================================================
# PART 7: APPLICATION USAGE
# ======================================================================

def process_transcript(transcript: str, visualize=True):
    """
    Process a transcript through both pipelines and evaluate the results.
    
    Args:
        transcript: The call transcript text
        visualize: Whether to create visualizations
        
    Returns:
        Tuple of (sequential_soap, multi_agent_soap, comparative_evaluation)
    """
    print("Processing transcript through sequential pipeline...")
    sequential_soap = run_sequential_pipeline(transcript)
    
    print("\nProcessing transcript through multi-agent system...")
    multi_agent_soap = run_multi_agent_system(transcript)
    
    print("\nEvaluating SOAP notes...")
    comparative_eval = evaluate_soap_notes(transcript, sequential_soap, multi_agent_soap)
    
    if visualize:
        print("\nGenerating visualizations...")
        visualize_comparative_evaluation(comparative_eval)
    
    print("\nGenerating text report...")
    report = generate_text_report(transcript, sequential_soap, multi_agent_soap, comparative_eval)
    print(report)
    
    return sequential_soap, multi_agent_soap, comparative_eval

# Example usage
def main():
    # Example transcript (authorization call about hip replacement)
    example_transcript = """
    Agent: Thank you for calling HealthFirst Customer Service. My name is Sarah. How can I help you today?

    Caller: Hi Sarah, I'm calling because my doctor wants me to have a hip replacement surgery next week, but he says I need to get it authorized first. I'm in a lot of pain and can barely walk.

    Agent: I'm sorry to hear about your pain. I'd be happy to help you with the authorization process. May I have your member ID and date of birth to access your account?

    Caller: Yes, my ID is ABC123456789 and my date of birth is January 15, 1955.

    Agent: Thank you. Let me pull up your information... I see you're enrolled in our Medicare Advantage plan. Can you tell me the name of your orthopedic surgeon and when the procedure is scheduled?

    Caller: My doctor is Dr. Johnson at Northside Orthopedics. The surgery is scheduled for next Tuesday, May 17th.

    Agent: Thank you. And do you know what CPT code Dr. Johnson is using for the procedure?

    Caller: I think he mentioned CPT 27130, but I'm not entirely sure.

    Agent: That sounds right for a total hip replacement. Has your doctor's office submitted any clinical documentation yet?

    Caller: I don't know. The nurse just told me to call and make sure it's approved before Tuesday. My pain is getting worse every day, and I'm worried about delaying the surgery.

    Agent: I understand your concern. Let me check our authorization system... I don't see a request from Dr. Johnson yet. For a hip replacement, we need updated X-rays from the last 30 days and a detailed letter of medical necessity explaining why the procedure is required.

    Caller: Oh no. So what should I do now? I really can't wait much longer.

    Agent: Don't worry. I'll initiate an authorization request right now and mark it as urgent given your situation. I'll also call Dr. Johnson's office directly to make sure they send over the required documentation today. Given the urgency, we can expedite the review process once we receive the documentation.

    Caller: Thank you so much. I really appreciate your help. This pain is unbearable, and I've been waiting for months to get this surgery.

    Agent: I completely understand. No one should have to wait in pain. I'll also provide you with a direct callback number in case you need to follow up. You should receive a decision within 24-48 hours once we receive all the documentation. Is there anything else I can help you with today?

    Caller: No, that's all. Thank you for your help.

    Agent: You're welcome. I hope your surgery goes well and you get relief soon. If you have any other questions, please don't hesitate to call back. Have a good day.
    """
    
    # Process the transcript
    sequential_soap, multi_agent_soap, comparative_eval = process_transcript(example_transcript)

if __name__ == "__main__":
    main()
