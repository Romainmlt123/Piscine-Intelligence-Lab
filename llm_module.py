import ollama

class LLMModule:
    def __init__(self, model="gemma:2b"):
        self.model = model
        print(f"LLM initialized with model: {self.model}")

    def generate_response(self, prompt, system_instruction="Tu es un professeur virtuel expert. Utilise le contexte fourni pour répondre précisément. Si la réponse n'est pas dans le contexte, dis-le."):
        try:
            messages = [
                {'role': 'system', 'content': system_instruction},
                {'role': 'user', 'content': prompt},
            ]
            response = ollama.chat(model=self.model, messages=messages)
            return response['message']['content']
        except Exception as e:
            print(f"Error generating response: {e}")
            return "Désolé, je ne peux pas répondre pour le moment."
