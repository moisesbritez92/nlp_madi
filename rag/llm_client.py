"""
LLM Client
==========
Client for interacting with Large Language Models via HTTP API.
Supports local/HTTP endpoints and OpenAI-compatible APIs.
"""

import requests
import os
from typing import List, Dict, Optional, Tuple
import time


class LLMClient:
    """
    Client for LLM inference.
    
    Supports:
    - Local HTTP endpoints (like Ollama)
    - OpenAI-compatible APIs
    """
    
    def __init__(
        self,
        mode: str = "local",
        url: str = "http://localhost:11434/api/chat",
        model: str = "gemma3",
        timeout: int = 120,
        openai_api_key: Optional[str] = None
    ):
        """
        Initialize LLM client.
        
        Args:
            mode: "local" for HTTP endpoint, "openai" for OpenAI API
            url: HTTP endpoint URL (for local mode)
            model: Model name
            timeout: Request timeout in seconds
            openai_api_key: OpenAI API key (for openai mode)
        """
        self.mode = mode
        self.url = url
        self.model = model
        self.timeout = timeout
        self.openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
    
    def call(
        self, 
        messages: List[Dict[str, str]]
    ) -> Tuple[str, float, Optional[str]]:
        """
        Call the LLM with a list of messages.
        
        Args:
            messages: List of {"role": "system"|"user"|"assistant", "content": "..."}
            
        Returns:
            Tuple of (response_text, time_seconds, error_message)
        """
        if self.mode == "openai":
            return self._call_openai(messages)
        else:
            return self._call_local(messages)
    
    def _call_local(
        self, 
        messages: List[Dict[str, str]]
    ) -> Tuple[str, float, Optional[str]]:
        """
        Call local HTTP endpoint (Ollama-style API).
        
        Args:
            messages: Chat messages
            
        Returns:
            Tuple of (response_text, time_seconds, error_message)
        """
        start_time = time.time()
        
        try:
            headers = {"Content-Type": "application/json"}
            
            payload = {
                "model": self.model,
                "stream": False,
                "messages": messages
            }
            
            response = requests.post(
                self.url,
                headers=headers,
                json=payload,
                timeout=self.timeout
            )
            
            response.raise_for_status()
            data = response.json()
            
            # Extract response content
            content = ""
            if "message" in data and "content" in data["message"]:
                content = data["message"]["content"]
            elif "response" in data:
                content = data["response"]
            elif "choices" in data and len(data["choices"]) > 0:
                # OpenAI-compatible format
                content = data["choices"][0].get("message", {}).get("content", "")
            
            elapsed = time.time() - start_time
            return content, elapsed, None
            
        except requests.exceptions.Timeout:
            return "", time.time() - start_time, f"Request timed out after {self.timeout}s"
        except requests.exceptions.ConnectionError:
            return "", time.time() - start_time, f"Could not connect to {self.url}"
        except requests.exceptions.HTTPError as e:
            return "", time.time() - start_time, f"HTTP error: {e}"
        except Exception as e:
            return "", time.time() - start_time, f"Error: {str(e)}"
    
    def _call_openai(
        self, 
        messages: List[Dict[str, str]]
    ) -> Tuple[str, float, Optional[str]]:
        """
        Call OpenAI API.
        
        Args:
            messages: Chat messages
            
        Returns:
            Tuple of (response_text, time_seconds, error_message)
        """
        start_time = time.time()
        
        if not self.openai_api_key:
            return "", 0.0, "OpenAI API key not configured"
        
        try:
            import openai
            
            client = openai.OpenAI(api_key=self.openai_api_key)
            
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                timeout=self.timeout
            )
            
            content = response.choices[0].message.content
            elapsed = time.time() - start_time
            
            return content, elapsed, None
            
        except ImportError:
            return "", 0.0, "openai package not installed. Run: pip install openai"
        except Exception as e:
            return "", time.time() - start_time, f"OpenAI error: {str(e)}"
    
    def test_connection(self) -> Tuple[bool, str]:
        """
        Test connection to the LLM endpoint.
        
        Returns:
            Tuple of (success, message)
        """
        try:
            test_messages = [
                {"role": "user", "content": "Say 'OK' if you can hear me."}
            ]
            
            response, elapsed, error = self.call(test_messages)
            
            if error:
                return False, error
            
            if response:
                return True, f"Connected successfully ({elapsed:.2f}s)"
            
            return False, "Empty response from model"
            
        except Exception as e:
            return False, str(e)


def get_default_llm_config() -> Dict:
    """
    Get default LLM configuration.
    
    Returns:
        Dict with default config values
    """
    return {
        "mode": "local",
        "url": "http://localhost:11434/api/chat",
        "model": "gemma3",
        "timeout": 120
    }
