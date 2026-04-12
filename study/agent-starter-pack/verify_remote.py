import asyncio
import vertexai
from vertexai import agent_engines


async def verify_async():
    project = "hupili-genai-bb"
    location = "us-central1"
    agent_id = "projects/160214671203/locations/us-central1/reasoningEngines/7216394430012456960"

    print("--- Verifying Deployed Agent (AgentEngine SDK - ASYNC) ---")
    print(f"ID: {agent_id}")

    vertexai.init(project=project, location=location)

    try:
        # Load the remote Agent Engine
        remote_app = agent_engines.get(agent_id)

        print(f"\nRemote App Type: {type(remote_app)}")
        print(
            f"Available Methods/Attributes: {[m for m in dir(remote_app) if not m.startswith('_')]}"
        )

        # Test 1: Try async_stream_query
        query = "Greet Pili using the hello_world tool"
        print(
            f"\nSending query via async_stream_query: message='{query}', user_id='test-pili'..."
        )
        try:
            # Check if it has the method
            if hasattr(remote_app, "async_stream_query"):
                responses = remote_app.async_stream_query(
                    message=query, user_id="test-pili"
                )
                print("\nResponses from Remote Agent (Async Streaming):")
                async for response in responses:
                    print(response)
                print("\n✅ Verification Successful!")
                return
            else:
                print("⚠️ 'async_stream_query' method not found on remote_app.")
        except Exception as e:
            print(f"⚠️ 'async_stream_query' failed: {e}")

        print("\n❌ Verification Failed: No supported query methods found.")

    except Exception as e:
        print(f"\n❌ Verification Failed: {e}")


if __name__ == "__main__":
    asyncio.run(verify_async())
