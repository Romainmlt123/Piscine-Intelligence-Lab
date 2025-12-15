import sys
from pathlib import Path

# Add parent to path for sibling imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.llm_module import LLMModule
from rag.rag_module import RAGModule
from config import KNOWLEDGE_BASE_DIR, LLM_MODEL_MATH, LLM_MODEL_PHYSICS, LLM_MODEL_ENGLISH, LLM_MODEL_GENERAL
import json
import time

class AgentOrchestrator:
    """
    Multi-model orchestrator with specialized LLMs per subject.
    
    Models:
    - MATH: qwen2.5:1.5b (configurable via LLM_MODEL_MATH)
    - PHYSICS: llama3.2:1b (configurable via LLM_MODEL_PHYSICS)
    - ENGLISH: gemma:2b (configurable via LLM_MODEL_ENGLISH)
    """
    
    def __init__(self):
        print("Initializing Orchestrator...")
        
        # Initialize Specialized LLMs per subject (using config)
        self.llm_math = LLMModule(model=LLM_MODEL_MATH)
        self.llm_physics = LLMModule(model=LLM_MODEL_PHYSICS)
        self.llm_english = LLMModule(model=LLM_MODEL_ENGLISH)
        self.llm_general = LLMModule(model=LLM_MODEL_GENERAL)
        
        # Initialize Specialized RAG Agents
        self.rag_math = RAGModule(collection_name="math_agent")
        self.rag_physics = RAGModule(collection_name="physics_agent")
        self.rag_english = RAGModule(collection_name="english_agent")
        
        # Ingest specific knowledge
        self.rag_math.ingest(str(KNOWLEDGE_BASE_DIR / "math"))
        self.rag_physics.ingest(str(KNOWLEDGE_BASE_DIR / "physics"))
        self.rag_english.ingest(str(KNOWLEDGE_BASE_DIR / "english"))
        
        print("Orchestrator initialized.")

    def route_query(self, text):
        """
        Uses Keywords first, then LLM to classify the query.
        Returns: 'MATH', 'PHYSICS', 'ENGLISH', or 'GENERAL'
        """
        text_lower = text.lower()
        
        # 1. Keyword Routing (Fast Path)
        math_keywords = ['équation', 'equation', 'racine', 'polynôme', 'polynome', 
                        'calcul', 'algèbre', 'algebra', 'math', 'x²', 'x^2', 
                        'dérivée', 'intégrale', 'fraction', 'nombre']
        physics_keywords = ['gravité', 'gravity', 'force', 'newton', 'mouvement', 
                           'énergie', 'energy', 'vitesse', 'accélération', 'physique',
                           'masse', 'poids', 'électricité', 'magnétisme']
        english_keywords = ['english', 'anglais', 'grammar', 'grammaire', 'vocabulary',
                           'tense', 'present', 'past', 'future', 'verb', 'conjugation',
                           'conjugaison', 'phrase', 'idiom', 'expression']
        
        if any(k in text_lower for k in math_keywords):
            print("Routing: Keyword Match -> MATH")
            return "MATH"
            
        if any(k in text_lower for k in physics_keywords):
            print("Routing: Keyword Match -> PHYSICS")
            return "PHYSICS"
            
        if any(k in text_lower for k in english_keywords):
            print("Routing: Keyword Match -> ENGLISH")
            return "ENGLISH"
            
        # 2. LLM Routing (Slow Fallback)
        print("Routing: No keyword match, falling back to LLM...")
        prompt = f"""
        Tu es un routeur intelligent. Analyse la demande suivante et classe-la dans une des catégories :
        - MATH (si ça parle d'équations, nombres, algèbre)
        - PHYSICS (si ça parle de forces, gravité, mouvement, énergie)
        - ENGLISH (si ça parle d'anglais, grammaire anglaise, vocabulaire anglais)
        - GENERAL (si c'est une conversation normale ou autre)
        
        Réponds UNIQUEMENT par un seul mot : MATH, PHYSICS, ENGLISH ou GENERAL.
        
        Demande : "{text}"
        """
        try:
            category = self.llm_general.generate_response(prompt, system_instruction="Tu es un classificateur strict. Réponds par un seul mot.").strip().upper()
            if "MATH" in category: return "MATH"
            if "PHYSICS" in category: return "PHYSICS"
            if "ENGLISH" in category: return "ENGLISH"
            return "GENERAL"
        except:
            return "GENERAL"

    def get_llm_for_subject(self, subject):
        """Returns the appropriate LLM instance for the subject"""
        if subject == "MATH":
            return self.llm_math
        elif subject == "PHYSICS":
            return self.llm_physics
        elif subject == "ENGLISH":
            return self.llm_english
        else:
            return self.llm_general

    def process(self, text):
        """
        Main pipeline: Route -> Retrieve -> Generate
        Returns: (response_text, source_name, agent_name, context, metrics)
        """
        metrics = {}
        
        # 1. Routing
        start_route = time.time()
        category = self.route_query(text)
        metrics['routing'] = time.time() - start_route
        
        # Get LLM for this subject
        llm = self.get_llm_for_subject(category)
        print(f"Routing decision: {category} -> Model: {llm.model_name} ({metrics['routing']:.2f}s)")
        
        context = ""
        source_name = ""
        agent_name = "Agent Général"
        system_prompt = "Tu es un assistant utile. Sois CONCIS."
        
        # 2. RAG Retrieval
        start_rag = time.time()
        if category == "MATH":
            agent_name = "Agent Maths"
            contexts, metadatas = self.rag_math.retrieve(text, n_results=3)
            if contexts:
                context = "\n\n".join(contexts)
                source_name = ", ".join([m.get('source', 'Inconnu') for m in metadatas])
            system_prompt = "Tu es un professeur de Mathématiques. Sois CONCIS. Phrases COURTES."
            
        elif category == "PHYSICS":
            agent_name = "Agent Physique"
            contexts, metadatas = self.rag_physics.retrieve(text, n_results=3)
            if contexts:
                context = "\n\n".join(contexts)
                source_name = ", ".join([m.get('source', 'Inconnu') for m in metadatas])
            system_prompt = "Tu es un professeur de Physique. Sois CONCIS. Phrases COURTES."
            
        elif category == "ENGLISH":
            agent_name = "Agent Anglais"
            contexts, metadatas = self.rag_english.retrieve(text, n_results=3)
            if contexts:
                context = "\n\n".join(contexts)
                source_name = ", ".join([m.get('source', 'Inconnu') for m in metadatas])
            system_prompt = "Tu es un professeur d'Anglais. Sois CONCIS. Phrases COURTES. Donne des exemples."
            
        metrics['rag'] = time.time() - start_rag
        print(f"RAG Retrieval: {metrics['rag']:.2f}s")
        
        # 3. LLM Generation
        if context:
            full_prompt = f"Contexte du cours :\n{context}\n\nQuestion : {text}"
        else:
            full_prompt = text
            
        start_llm = time.time()
        response_text = llm.generate_response(full_prompt, system_instruction=system_prompt)
        metrics['llm'] = time.time() - start_llm
        print(f"LLM Generation: {metrics['llm']:.2f}s")
        
        return response_text, source_name, agent_name, context, metrics

    def process_stream(self, text):
        """
        Generator that yields:
        - ('routing', {'agent': agent_name, 'model': model_name})
        - ('rag', {context, source})
        - ('llm_chunk', token)
        - ('metrics', metrics_dict)
        """
        metrics = {'stt': 0, 'routing': 0, 'rag': 0, 'llm': 0, 'tts': 0, 'total': 0}
        
        # 1. Routing
        start_routing = time.time()
        subject = self.route_query(text)
        metrics['routing'] = time.time() - start_routing
        
        # Get LLM for this subject
        llm = self.get_llm_for_subject(subject)
        model_name = llm.model_name
        
        # Set agent name based on subject
        if subject == "MATH":
            agent_name = "MATH"
        elif subject == "PHYSICS":
            agent_name = "PHYSICS"
        elif subject == "ENGLISH":
            agent_name = "ENGLISH"
        else:
            agent_name = "GENERAL"
            
        yield ('routing', {'agent': agent_name, 'model': model_name})
        
        # 2. RAG Retrieval
        start_rag = time.time()
        context = ""
        source_name = "Aucune source"
        
        if subject == "MATH":
            context, metadata = self.rag_math.retrieve(text)
            system_prompt = "Tu es un professeur de Mathématiques. Sois CONCIS. Utilise des phrases COURTES. Va droit au but."
        elif subject == "PHYSICS":
            context, metadata = self.rag_physics.retrieve(text)
            system_prompt = "Tu es un professeur de Physique. Sois CONCIS. Utilise des phrases COURTES. Va droit au but."
        elif subject == "ENGLISH":
            context, metadata = self.rag_english.retrieve(text)
            system_prompt = "Tu es un professeur d'Anglais. Sois CONCIS. Utilise des phrases COURTES. Donne des exemples."
        else:
            context = ""
            metadata = {}
            system_prompt = "Tu es un assistant. Sois CONCIS. Utilise des phrases COURTES. Va droit au but."
            
        if context:
            # Handle list of sources if multiple chunks
            if isinstance(metadata, list):
                source_name = ", ".join([m.get('source', 'Inconnu') for m in metadata])
            else:
                source_name = metadata.get('source', 'Inconnu')
        
        metrics['rag'] = time.time() - start_rag
        yield ('rag', {'context': context, 'source': source_name})
        
        # 3. LLM Generation (Streaming)
        if context:
            full_prompt = f"Contexte du cours :\n{context}\n\nQuestion : {text}"
        else:
            full_prompt = text
            
        start_llm = time.time()
        
        # Stream tokens using the subject-specific LLM
        for chunk in llm.generate_response_stream(full_prompt, system_instruction=system_prompt):
            yield ('llm_chunk', chunk)
            
        metrics['llm'] = time.time() - start_llm
        metrics['model'] = model_name
        yield ('metrics', metrics)

