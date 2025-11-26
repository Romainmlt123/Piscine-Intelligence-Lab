import ollama

class LLMModule:
    def __init__(self, model="gemma:2b"):
        self.model = model
        print(f"LLM initialized with model: {self.model}")

    def generate_response(self, prompt):
        try:
            response = ollama.chat(model=self.model, messages=[
                {
                    'role': 'user',
                    'content': prompt,
                },
            ])
            return response['message']['content']
        except Exception as e:
            print(f"Error generating response: {e}")
            return "Désolé, je ne peux pas répondre pour le moment."
