import openai

# Configure the client to point to our llb load balancer
client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"
)

print("--- Testing: Getting all models from the load balancer ---")
try:
    models_response = client.models.list()
    model_ids = [model.id for model in models_response.data]
    
    if not model_ids:
        print("❌ Test Failed: Load balancer returned no models.")
        print("   Check if your LMStudio server is running and if the llb-monitor has found it.")
    else:
        print("✅ Test Passed: Successfully retrieved models.")
        print("Available models:", model_ids)
        
        # Select the first available model for the chat completion test
        selected_model = model_ids[0]
        
        print(f"\n--- Testing: Chat completion with model: {selected_model} ---")
        
        try:
            stream = client.chat.completions.create(
                model=selected_model,
                messages=[{"role": "user", "content": "What is the capital of France?"}],
                stream=True,
            )
            
            print("✅ Test Passed: Stream started. Response:")
            full_response = ""
            for chunk in stream:
                content = chunk.choices[0].delta.content
                if content:
                    print(content, end="", flush=True)
                    full_response += content
            
            if not full_response:
                print("\n❌ Test Warning: Stream completed but no content was received.")

        except Exception as e:
            print(f"\n❌ Test Failed: Could not get chat completion.")
            print(f"   Error: {e}")

except openai.APIConnectionError as e:
    print("❌ Test Failed: Could not connect to the llb load balancer.")
    print("   Please ensure the docker-compose stack is running.")
    print(f"   Error: {e.__cause__}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
