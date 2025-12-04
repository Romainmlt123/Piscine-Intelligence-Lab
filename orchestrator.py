from llm_module import LLMModule
from rag_module import RAGModule
import json
import time

class AgentOrchestrator:
    def __init__(self):
        print("Initializing Orchestrator...")
        self.llm = LLMModule(model="qwen2.5:1.5b")
        
        # Initialize Specialized Agents (RAG)
        self.rag_math = RAGModule(collection_name="math_agent")
        self.rag_physics = RAGModule(collection_name="physics_agent")
        
        # Ingest specific knowledge
        self.rag_math.ingest("./knowledge_base/math")
        self.rag_physics.ingest("./knowledge_base/physics")
        
        print("Orchestrator initialized.")

    def route_query(self, text):
        """
        Uses Keywords first, then LLM to classify the query.
        Returns: 'MATH', 'PHYSICS', or 'GENERAL'
        """
        text_lower = text.lower()
        
        # 1. Keyword Routing (Fast Path)
        math_keywords = ['équation', 'equation', 'racine', 'polynôme', 'polynome', 'calcul', 'algèbre', 'algebra', 'math', 'x²', 'x^2']
        physics_keywords = ['gravité', 'gravity', 'force', 'newton', 'mouvement', 'énergie', 'energy', 'vitesse', 'accélération', 'physique']
        
        if any(k in text_lower for k in math_keywords):
            print("Routing: Keyword Match -> MATH")
            return "MATH"
            
        if any(k in text_lower for k in physics_keywords):
            print("Routing: Keyword Match -> PHYSICS")
            return "PHYSICS"
            
        # 2. LLM Routing (Slow Fallback)
        print("Routing: No keyword match, falling back to LLM...")
        prompt = f"""
        Tu es un routeur intelligent. Analyse la demande suivante et classe-la dans une des catégories :
        - MATH (si ça parle d'équations, nombres, algèbre)
        - PHYSICS (si ça parle de forces, gravité, mouvement, énergie)
        - GENERAL (si c'est une conversation normale ou autre)
        
        Réponds UNIQUEMENT par un seul mot : MATH, PHYSICS ou GENERAL.
        
        Demande : "{text}"
        """
        try:
            category = self.llm.generate_response(prompt, system_instruction="Tu es un classificateur strict. Réponds par un seul mot.").strip().upper()
            if "MATH" in category: return "MATH"
            if "PHYSICS" in category: return "PHYSICS"
            return "GENERAL"
        except:
            return "GENERAL"

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
        print(f"Routing decision: {category} ({metrics['routing']:.2f}s)")
        
        context = ""
        source_name = ""
        agent_name = "Agent Général"
        system_prompt = "Tu es un assistant utile."
        
        # 2. RAG Retrieval
        start_rag = time.time()
        if category == "MATH":
            agent_name = "Agent Maths"
            contexts, metadatas = self.rag_math.retrieve(text, n_results=3)
            if contexts:
                context = "\n\n".join(contexts)
                source_name = ", ".join([m.get('source', 'Inconnu') for m in metadatas])
            system_prompt = "Tu es un professeur de Mathématiques rigoureux. Utilise le contexte pour répondre."
            
        elif category == "PHYSICS":
            agent_name = "Agent Physique"
            contexts, metadatas = self.rag_physics.retrieve(text, n_results=3)
            if contexts:
                context = "\n\n".join(contexts)
                source_name = ", ".join([m.get('source', 'Inconnu') for m in metadatas])
            system_prompt = "Tu es un professeur de Physique passionné. Utilise des exemples concrets et le contexte."
        metrics['rag'] = time.time() - start_rag
        print(f"RAG Retrieval: {metrics['rag']:.2f}s")
        
        # 3. LLM Generation
        if context:
            full_prompt = f"Contexte du cours :\n{context}\n\nQuestion : {text}"
        else:
            full_prompt = text
            
        start_llm = time.time()
        response_text = self.llm.generate_response(full_prompt, system_instruction=system_prompt)
        metrics['llm'] = time.time() - start_llm
        print(f"LLM Generation: {metrics['llm']:.2f}s")
        
        return response_text, source_name, agent_name, context, metrics
