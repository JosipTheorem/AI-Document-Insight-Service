"""Docker startup must use service URLs and prepare only the published corpus."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from config import get_model, model_name
from config import EMBEDDING_MODEL
from deployment.bootstrap import main
from evaluation.runner import make_judge


class DeploymentTests(unittest.TestCase):
    def test_agent_and_judge_use_the_configured_ollama_endpoint(self):
        with patch("config.OLLAMA_URL", "http://ollama:11434"):
            self.assertEqual(get_model().base_url, "http://ollama:11434")
        with patch("evaluation.runner.OLLAMA_URL", "http://ollama:11434"):
            self.assertEqual(make_judge().base_url, "http://ollama:11434")

    @patch("deployment.bootstrap.log_event")
    @patch("builtins.print")
    @patch("deployment.bootstrap.Index")
    @patch("deployment.bootstrap.Client")
    def test_startup_downloads_only_missing_models_and_indexes_six_examples(self, client, index, *_):
        client.return_value.list.return_value.models = [SimpleNamespace(model=model_name)]
        client.return_value.pull.return_value = iter([{"status": "success"}])
        index.return_value.ingest.return_value = {"filename": "test.pdf", "status": "indexed"}
        index.return_value.documents.return_value = [None] * 6
        main()
        client.return_value.pull.assert_called_once_with(EMBEDDING_MODEL, stream=True)
        paths = [call.args[0] for call in index.return_value.ingest.call_args_list]
        self.assertEqual(len(paths), 6)
        self.assertEqual(sum(path.suffix == ".pdf" for path in paths), 4)
        self.assertEqual(sum(path.suffix == ".png" for path in paths), 2)
        self.assertTrue(all(path.is_file() for path in paths))
        self.assertEqual(paths[0].name, "kazneni_zakon_izvadak.pdf")

    @patch("deployment.bootstrap.log_event")
    @patch("builtins.print")
    @patch("deployment.bootstrap.Index")
    @patch("deployment.bootstrap.Client")
    def test_restart_reuses_installed_models_and_stops_on_download_failure(self, client, index, *_):
        client.return_value.list.return_value.models = [SimpleNamespace(model=name)
                                                       for name in (model_name, EMBEDDING_MODEL)]
        index.return_value.ingest.return_value = {"filename": "test.pdf", "status": "unchanged"}
        main()
        client.return_value.pull.assert_not_called()
        index.reset_mock()
        client.return_value.list.return_value.models = []
        client.return_value.pull.side_effect = ConnectionError("download failed")
        with self.assertRaises(ConnectionError):
            main()
        index.assert_not_called()


if __name__ == "__main__":
    unittest.main()
