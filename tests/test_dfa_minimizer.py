"""Проверки минимизации, строгого CSV и автономного CLI."""

import csv
import itertools
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest

import dfa_minimizer as app


ROOT = Path(__file__).resolve().parents[1]


def accepts(dfa: app.DFA, word: tuple[str, ...]) -> bool:
    """Независимо проверить слово проходом по переходам."""
    state = dfa.initial
    for symbol in word:
        state = dfa.transitions[state][dfa.alphabet.index(symbol)]
    return state in dfa.accepting


def distinguishable(dfa: app.DFA, left: str, right: str) -> bool:
    """Найти различающее слово обходом произведения состояний."""
    pending = [(left, right)]
    seen = set(pending)
    for a, b in pending:
        if (a in dfa.accepting) != (b in dfa.accepting):
            return True
        for pair in zip(dfa.transitions[a], dfa.transitions[b]):
            if pair not in seen:
                seen.add(pair)
                pending.append(pair)
    return False


class MinimizationTests(unittest.TestCase):
    def test_subset_normalization(self):
        for text in ("{AD}", "{A,D}", "{ D, A }"):
            self.assertEqual(app.normalize_state(text), "{A,D}")
        self.assertEqual(app.normalize_state("{q1,q0}"), "{q0,q1}")
        self.assertEqual(app.normalize_state("{}"), "{}")
        self.assertEqual(app.normalize_state("{{D},{C,B}}"), "{{B,C},{D}}")
        for text in ("{A,A}", "{AA}", "{A,}", "{A", "A}", "{A,{B}}", "{{A}"):
            with self.subTest(text=text), self.assertRaises(app.InputError):
                app.normalize_state(text)

    def test_subset_minimization_and_roundtrip(self):
        dfa = app.read_dfa(ROOT / "examples" / "dfa_subsets.csv")
        result = app.minimize(dfa).automaton
        self.assertEqual(result.states, ("{A,D}", "{{B,C},{D}}", "{C}"))
        self.assertEqual(result.accepting, frozenset({"{{B,C},{D}}"}))
        for length in range(6):
            for word in itertools.product(dfa.alphabet, repeat=length):
                self.assertEqual(accepts(dfa, word), accepts(result, word))
        with tempfile.TemporaryDirectory() as directory:
            output = app.save_rows(Path(directory), "result", app.dfa_rows(result))
            restored = app.read_dfa(output)
            self.assertEqual(restored, result)
            self.assertEqual(app.minimize(restored).automaton, result)

    def test_subset_aliases_duplicates_and_empty_subset(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.csv"
            path.write_text('состояние,начальное,допускающее,a\n{AD},1,1,"{D,A}"\n', encoding="utf-8")
            dfa = app.read_dfa(path)
            self.assertEqual(dfa.transitions, {"{A,D}": ("{A,D}",)})
            with path.open("a", encoding="utf-8") as stream:
                stream.write('"{A,D}",0,1,{AD}\n')
            with self.assertRaises(app.InputError):
                app.read_dfa(path)
            path.write_text('состояние,начальное,допускающее,a\n{A},1,1,{}\n{},0,0,{}\n', encoding="utf-8")
            self.assertEqual(app.minimize(app.read_dfa(path)).automaton.states, ("{A}", "{}"))

    def test_subset_cli(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.csv"
            path.write_bytes((ROOT / "examples" / "dfa_subsets.csv").read_bytes())
            process = subprocess.run([sys.executable, str(ROOT / "dfa_minimizer.py")],
                                     input=f"{path}\n\n\n", text=True, capture_output=True, timeout=10)
            self.assertEqual(process.returncode, 0, process.stderr)
            self.assertIn("{{B,C},{D}}", process.stdout)
            output = app.read_dfa(path.parent / "dfa_minimized.csv")
            self.assertEqual(len(output.states), 3)

    def test_merge_and_remove(self):
        dfa = app.DFA(("s", "a", "b", "lost"), ("0", "1"), "s", frozenset({"a", "b"}),
                      {"s": ("a", "b"), "a": ("a", "b"), "b": ("a", "b"),
                       "lost": ("lost", "lost")})
        result = app.minimize(dfa)
        self.assertEqual(len(result.automaton.states), 2)
        self.assertEqual(result.unreachable, ("lost",))
        self.assertIn(("a", "b"), result.groups.values())

    def test_degenerate_automata(self):
        for alphabet in ((), ("a",)):
            for accepting in (frozenset(), frozenset({"s", "t"})):
                dfa = app.DFA(("s", "t"), alphabet, "s", accepting,
                              {"s": ("t",) if alphabet else (), "t": ("s",) if alphabet else ()})
                self.assertEqual(len(app.minimize(dfa).automaton.states), 1)

    def test_seeded_automata_language_and_minimality(self):
        rng = random.Random(17)
        for _ in range(60):
            states = tuple(f"s{i}" for i in range(rng.randint(1, 8)))
            dfa = app.DFA(states, ("a", "b"), states[0],
                          frozenset(s for s in states if rng.choice((True, False))),
                          {s: (rng.choice(states), rng.choice(states)) for s in states})
            result = app.minimize(dfa).automaton
            for length in range(6):
                for word in itertools.product(dfa.alphabet, repeat=length):
                    self.assertEqual(accepts(dfa, word), accepts(result, word))
            for left, right in itertools.combinations(result.states, 2):
                self.assertTrue(distinguishable(result, left, right))

    def test_csv_errors_and_bom(self):
        cases = ["", "состояние,начальное,допускающее\ns,0,0\n",
                 "состояние,начальное,допускающее,a\ns,1,0,unknown\n",
                 "состояние,начальное,допускающее,a,a\ns,1,0,s,s\n",
                 "состояние,начальное,допускающее\ns,1,0\ns,0,1\n",
                 "состояние,начальное,допускающее\ns,1,0,лишнее\n",
                 'состояние,начальное,допускающее\n"s,1,0\n',
                 "состояние,начальное,допускающее\ns,1,0\n\n"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.csv"
            for text in cases:
                with self.subTest(text=text):
                    path.write_text(text, encoding="utf-8")
                    with self.assertRaises((app.InputError, csv.Error)):
                        app.read_dfa(path)
            path.write_text("\ufeffсостояние,начальное,допускающее\ns,1,1\n", encoding="utf-8")
            self.assertEqual(app.read_dfa(path).initial, "s")

    def test_cli_and_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.csv"
            path.write_bytes((ROOT / "examples" / "dfa.csv").read_bytes())
            for suffix in ("", "_1"):
                process = subprocess.run([sys.executable, str(ROOT / "dfa_minimizer.py")],
                                         input=f"{path}\n\n\n", text=True, capture_output=True, timeout=10)
                self.assertEqual(process.returncode, 0, process.stderr)
                output = path.parent / f"dfa_minimized{suffix}.csv"
                self.assertIn(str(output), process.stdout)
                self.assertEqual(len(app.read_dfa(output).states), 3)


if __name__ == "__main__":
    unittest.main()