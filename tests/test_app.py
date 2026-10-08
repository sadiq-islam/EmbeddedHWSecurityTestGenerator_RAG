"""Current UI regression: process, change/remove PDF, and invalidate chunk settings."""
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
from streamlit.testing.v1 import AppTest
from test_workflow import Embedder, chunk, FakeClient, Counter
from src.chatbot import HardwarePlanner


class InterfaceTests(unittest.TestCase):
    def test_processing_and_document_removal_clear_old_state(self):
        # Mock payloads to simulate user interactions without requiring actual file I/O
        upload = NS(name="manual.pdf", getvalue=lambda: b"synthetic-pdf-A")
        fake_doc = NS(pages={1: object()}, tables=[])
        planner = HardwarePlanner(client=FakeClient(), counter=Counter())

        # Isolate the application from all heavy ML models and network dependencies
        with patch("streamlit.file_uploader", return_value=upload) as uploader, \
             patch("src.vector_store.load_embedder", return_value=Embedder()), \
             patch("src.pdf_processor.PDFProcessor.extract_document",
                   return_value=(fake_doc, [chunk()], 1)), \
             patch("src.chatbot.HardwarePlanner", return_value=planner):

            # Boot up the Streamlit test application instance
            app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"),
                                    default_timeout=20).run()
            self.assertFalse(app.exception)

            # Simulate basic UI interactions: filling API key and triggering processing
            app.text_input[0].set_value("test-key").run()
            app.button[0].click().run()
            self.assertFalse(app.exception)

            # Verify the index was successfully built and attached to session state
            self.assertIsNotNone(app.session_state["vector_store"])
            self.assertEqual([m.value for m in app.metric], ["1 / 1", "0", "1"])

            # Seed the interface with conversational state to test cache invalidation
            app.session_state["current_draft"] = "old draft"
            app.session_state["chat_messages"] = [{"role": "user", "content": "old"}]
            planner.chat_history = [{"role": "user", "content": "old"}]

            # Simulate a user swapping the active document for a new one
            uploader.return_value = NS(name="same-name.pdf", getvalue=lambda: b"synthetic-pdf-B")
            app.run()
            self.assertFalse(app.exception)

            # CRITICAL SECURITY/STATE CHECK: Ensure switching documents purges the old vector store and history
            self.assertIsNone(app.session_state["vector_store"])
            self.assertEqual(app.session_state["current_draft"], "")
            self.assertEqual(app.session_state["chat_messages"], [])
            self.assertEqual(planner.chat_history, [])

            app.button[0].click().run()

            # Simulate a user clearing the file uploader entirely
            uploader.return_value = None
            app.run()
            self.assertFalse(app.exception)
            self.assertIsNone(app.session_state["vector_store"])


if __name__ == "__main__":
    unittest.main()