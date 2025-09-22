import sys
from pathlib import Path
# Ensure repository root is on sys.path for imports
sys.path.append(str(Path(__file__).resolve().parents[1]))

import asyncio
# Load langgraph_orchestrator module directly from file to avoid triggering package __init__
import importlib.util
module_path = Path(__file__).resolve().parents[1] / "muse_v3_advanced" / "langgraph_orchestrator.py"
spec = importlib.util.spec_from_file_location("langgraph_orchestrator", str(module_path))
lang_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lang_mod)
LangGraphOrchestrator = lang_mod.LangGraphOrchestrator

# Minimal dummy octotools framework and components for testing
class DummyTool:
    def __init__(self, name):
        self.name = name
    async def execute(self, **kwargs):
        # Return a mock successful result structure
        return {"success": True, "result": {"success": True, "results": [{"title": "Mock Item", "price": 99}]}}

class DummyOcto:
    def __init__(self):
        self._tools = {name: DummyTool(name) for name in ["search","recommend","compare","filter","translate","visual_search"]}
    def create_plan(self, **kwargs):
        class Plan:
            def __init__(self):
                self.plan = "noop"
        return Plan()
    @property
    def tools(self):
        return self._tools

class DummyComponent:
    pass

components = {
    "text_encoder": None,
    "image_encoder": None,
    "metadata_encoder": None,
    "fusion_layer": None,
    "intent_classifier": None,
    "state_tracker": None,
    "tool_selector": None,
    "octotools_framework": DummyOcto(),
    "response_generator": None
}

orch = LangGraphOrchestrator(components)

async def run_test():
    # Provide an input that includes an image to exercise perception multimodal path
    session_context = {"images": [{"id": "img1", "data": "mock"}], "conversation_history": [], "current_turn": 1, "user_profile": {}}
    res = await orch.process_conversation("Show me dresses similar to this image that are suitable for office wear.", session_context=session_context)

    print("=== Perception Block Test Result ===")
    print("success:", res.get("success"))
    print("intent:", res.get("intent"))
    print("language:", res.get("language"))
    print("processing_time:", res.get("processing_time"))
    print("multimodal_elements:", res.get("multimodal_elements"))
    # print debug info if available
    if res.get("debug_info"):
        print("debug_info:", res.get("debug_info"))

if __name__ == "__main__":
    asyncio.run(run_test())
