# -*- coding: utf-8 -*-
import unittest

from cosmian_ai_runner.app import app


class ApiTest(unittest.TestCase):
    def setUp(self):
        app.testing = True
        self.app = app.test_client()

    def test_health(self):
        res = self.app.get("/health")
        self.assertEqual(res.status_code, 200)

    def test_summarize_no_doc(self):
        res = self.app.post("/summarize")
        self.assertEqual(res.status_code, 400)

    def test_summarize_short_text(self):
        res = self.app.post("/summarize", data={"doc": "hello"})
        self.assertEqual(res.status_code, 400)

    def test_summarize(self):
        res = self.app.post(
            "/summarize",
            data={"doc": "Hello " * 50},
        )
        self.assertEqual(res.status_code, 200)
        self.assertIn("summary", res.json)

    def test_translate_no_doc(self):
        res = self.app.post("/translate")
        self.assertEqual(res.status_code, 400)

    def test_translate_no_lang(self):
        res = self.app.post(
            "/translate",
            data={"doc": "Hello"},
        )
        self.assertEqual(res.status_code, 400)

    def test_translate_wrong_lang(self):
        res = self.app.post(
            "/translate",
            data={"doc": "Hello", "src_lang": "Wrong", "tgt_lang": "Missing"},
        )
        self.assertEqual(res.status_code, 400)

    def test_translate(self):
        res = self.app.post(
            "/translate",
            data={"doc": "Hello", "src_lang": "en", "tgt_lang": "fr"},
        )
        self.assertEqual(res.status_code, 200)
        self.assertIn("translation", res.json)


if __name__ == "__main__":
    unittest.main()
