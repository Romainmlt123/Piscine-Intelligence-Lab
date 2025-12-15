import ollama

class LLMModule:
    def __init__(self, model="qwen2.5:1.5b"):
        self.model = model
        self.model_name = model  # For display purposes
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

    def generate_response_stream(self, prompt, system_instruction="Tu es un professeur virtuel expert."):
        """
        Yields chunks of text as they are generated.
        """
        try:
            messages = [
                {'role': 'system', 'content': system_instruction},
                {'role': 'user', 'content': prompt},
            ]
            stream = ollama.chat(model=self.model, messages=messages, stream=True)
            for chunk in stream:
                content = chunk['message']['content']
                yield content
        except Exception as e:
            print(f"Error generating stream: {e}")
            yield f"Error: {e}"

