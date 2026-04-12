import vertexai
from vertexai.preview.reasoning_engines import ReasoningEngine


def verify():
    project = "hupili-genai-bb"
    location = "us-central1"
    agent_id = "projects/160214671203/locations/us-central1/reasoningEngines/7216394430012456960"

    print("--- Verifying Deployed Agent (ReasoningEngine SDK) ---")
    print(f"ID: {agent_id}")

    vertexai.init(project=project, location=location)

    try:
        # Load the remote Reasoning Engine
        remote_app = ReasoningEngine(agent_id)

        print(f"\nRemote App Type: {type(remote_app)}")
        print(
            f"Available Methods/Attributes: {[m for m in dir(remote_app) if not m.startswith('_')]}"
        )
        print(f"Operation Schemas: {remote_app.operation_schemas()}")

        print("\nSending query via query: input='Greet Pili'...")
        try:
            # According to ReasoningEngine API, it should have a 'query' method
            # or be callable if it wrapped the class correctly.
            response = remote_app.query(input="Greet Pili")
            print(f"\nResponse from Remote Agent: {response}")
            print("\n✅ Verification Successful!")
            return
        except Exception as e:
            print(f"⚠️ 'query' failed: {e}")

        print("\nTrying direct call...")
        try:
            response = remote_app(input="Greet Pili")
            print(f"\nResponse from Remote Agent (Call): {response}")
            print("\n✅ Verification Successful!")
            return
        except Exception as e:
            print(f"⚠️ direct call failed: {e}")

        print("\n❌ Verification Failed: No query methods found.")

    except Exception as e:
        print(f"\n❌ Verification Failed: {e}")


if __name__ == "__main__":
    verify()
