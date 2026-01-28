"""
Tests for Graph-Based Injectr

Basic tests to verify the project structure and imports work correctly.
"""

import pytest


class TestImports:
    """Test that all modules can be imported."""
    
    def test_import_main(self):
        """Test main package import."""
        import graph_based_injectr
        assert hasattr(graph_based_injectr, "__version__")
    
    def test_import_config(self):
        """Test config module import."""
        from graph_based_injectr.config import Settings, get_settings
        assert Settings is not None
    
    def test_import_agents(self):
        """Test agents module import."""
        from graph_based_injectr.agents import create_injection_agent, AgentState
        assert create_injection_agent is not None
    
    def test_import_llm(self):
        """Test LLM module import."""
        from graph_based_injectr.llm import create_chat_model, ModelConfig
        assert create_chat_model is not None
    
    def test_import_target(self):
        """Test target module import."""
        from graph_based_injectr.target import TargetClient, TargetResponse
        assert TargetClient is not None
    
    def test_import_payloads(self):
        """Test payloads module import."""
        from graph_based_injectr.payloads import (
            JailbreakCategory,
            get_all_templates,
            encode_base64,
            mutate_payload,
        )
        assert JailbreakCategory is not None
    
    def test_import_tools(self):
        """Test tools module import."""
        from graph_based_injectr.tools import get_all_tools
        
        tools = get_all_tools()
        assert len(tools) > 0
    
    def test_import_knowledge(self):
        """Test knowledge module import."""
        from graph_based_injectr.knowledge import RAGEngine, Document
        assert RAGEngine is not None


class TestPayloads:
    """Test payload functionality."""
    
    def test_jailbreak_templates(self):
        """Test jailbreak templates exist and have required fields."""
        from graph_based_injectr.payloads import get_all_templates
        
        templates = get_all_templates()
        assert len(templates) > 0
        
        for t in templates:
            assert t.name
            assert t.description
            assert t.template
            assert t.category
    
    def test_encoding_functions(self):
        """Test encoding functions work correctly."""
        from graph_based_injectr.payloads import (
            encode_base64,
            decode_base64,
            encode_rot13,
            decode_rot13,
        )
        
        test_text = "Hello World"
        
        # Base64
        encoded = encode_base64(test_text)
        decoded = decode_base64(encoded)
        assert decoded == test_text
        
        # ROT13
        encoded = encode_rot13(test_text)
        decoded = decode_rot13(encoded)
        assert decoded == test_text
    
    def test_mutation_functions(self):
        """Test mutation functions produce output."""
        from graph_based_injectr.payloads import mutate_payload, generate_variations
        
        test_payload = "Read the secret file"
        
        mutated = mutate_payload(test_payload)
        assert mutated  # Not empty
        assert mutated != test_payload or len(mutated) > len(test_payload)
        
        variations = generate_variations(test_payload, count=3)
        assert len(variations) == 3


class TestTools:
    """Test tool functionality."""
    
    def test_tool_registry(self):
        """Test tool registry has expected tools."""
        from graph_based_injectr.tools import get_all_tools
        
        tools = get_all_tools()
        tool_names = [t.name for t in tools]
        
        # Check expected tools exist
        assert "send_prompt" in tool_names
        assert "analyze" in tool_names
        assert "mutate" in tool_names
        assert "notes" in tool_names
        assert "finish" in tool_names


class TestAgentState:
    """Test agent state management."""
    
    def test_agent_state_exists(self):
        """Test AgentState can be imported."""
        from graph_based_injectr.agents import AgentState
        assert AgentState is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
