"""Run: python3 -m unittest webapp/test_app.py  (no Claude calls: `ask` is faked)."""
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).parent))
import app  # noqa: E402
from app import score  # noqa: E402


class Score(unittest.TestCase):
    def test_whole_words_case_insensitive(self):
        r = score("Built ETL in Python and SQL on AWS.", ["python", "SQL", "Java", "AWS"])
        self.assertEqual(r["score"], 75)
        self.assertEqual(r["missing"], ["Java"])

    def test_no_partial_word_hits(self):
        # "Java" must not match "JavaScript"; "C++" must match despite the regex characters.
        r = score("JavaScript and C++ developer", ["Java", "C++"])
        self.assertEqual((r["matched"], r["missing"]), (["C++"], ["Java"]))

    def test_duplicates_and_empty(self):
        self.assertEqual(score("go", ["Go", "go", " "])["score"], 100)
        self.assertEqual(score("anything", [])["score"], 0)


BODY = {"profile": {"resume_text": "Python, SQL and Tableau dashboards."},
        "job": {"title": "DE", "company": "X", "summary": "", "apply_url": "https://x", "keywords": ["Python", "SQL", "Tableau", "Airflow"]}}


class TailorNeverScoresLower(unittest.TestCase):
    def run_with(self, *answers):
        with mock.patch.object(app, "ask", side_effect=list(answers)) as fake:
            return app.agent_tailor(BODY), fake.call_count

    def test_dropped_keyword_is_restored(self):
        r, calls = self.run_with({"resume_markdown": "Python and SQL.", "changes": ["cut"]},
                                 {"resume_markdown": "Python, SQL, Tableau.", "changes": ["restored Tableau"]})
        self.assertEqual((calls, r["before"]["score"], r["after"]["score"], r["lost"]), (2, 75, 75, []))
        self.assertEqual(r["repaired"], ["Tableau"])

    def test_original_kept_when_repair_fails(self):
        r, _ = self.run_with({"resume_markdown": "Python.", "changes": ["cut"]},
                             {"resume_markdown": "Python.", "changes": []})
        self.assertEqual(r["resume_markdown"], BODY["profile"]["resume_text"])
        self.assertEqual(r["after"]["score"], r["before"]["score"])
        self.assertTrue(r["note"])

    def test_no_repair_call_when_nothing_lost(self):
        r, calls = self.run_with({"resume_markdown": "Python, SQL, Tableau, Airflow.", "changes": ["added Airflow"]})
        self.assertEqual((calls, r["after"]["score"], r["gained"]), (1, 100, ["Airflow"]))


if __name__ == "__main__":
    unittest.main()
