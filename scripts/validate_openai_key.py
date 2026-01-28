import os
import sys
from typing import Optional
import openai
from dotenv import load_dotenv

# Add the project root to sys.path so we can import graph_based_injectr
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from graph_based_injectr.config import get_settings
except ImportError:
    # Fallback if the script is run in a way that graph_based_injectr isn't found
    get_settings = None

def check_openai_key(api_key: Optional[str] = None) -> bool:
    """
    Checks if an OpenAI API key is valid by making a simple request to list models.
    
    Args:
        api_key: The OpenAI API key to check. If None, it will try to load from 
                environment or project settings.
    
    Returns:
        bool: True if valid, False otherwise.
    """
    # Load environment variables from .env if present
    load_dotenv()
    
    if not api_key:
        if get_settings:
            try:
                settings = get_settings()
                api_key = settings.openai_api_key
            except Exception:
                api_key = os.getenv("OPENAI_API_KEY")
        else:
            api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        print("Error: No OpenAI API key provided or found in environment/settings.")
        print("Usage: python validate_openai_key.py [YOUR_API_KEY]")
        return False

    print(f"Checking key starting with: {api_key[:8]}...")
    
    client = openai.OpenAI(api_key=api_key)
    
    try:
        # A simple request that doesn't cost tokens (usually)
        client.models.list()
        print("✅ Success: OpenAI API key is valid.")
        return True
    except openai.AuthenticationError:
        print("❌ Error: Invalid OpenAI API key (AuthenticationError).")
        return False
    except openai.RateLimitError:
        print("✅ Success: OpenAI API key is valid, but you are being rate limited.")
        return True
    except openai.OpenAIError as e:
        print(f"❌ Error: An OpenAI error occurred: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: An unexpected error occurred: {e}")
        return False

if __name__ == "__main__":
    # Allow passing key as command line argument
    key_arg = sys.argv[1] if len(sys.argv) > 1 else None
    
    if check_openai_key(key_arg):
        sys.exit(0)
    else:
        sys.exit(1)
