"""Проверки синтаксиса и решения независимым ограниченным языковым вычислением."""

from concurrent.futures import ThreadPoolExecutor
import csv
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import dfa_minimizer
import regular_equations_solver as app


ROOT = Path(__file__).resolve().parents[1]
LIMIT = 4


def product(left: set[str], right: set[str]) -> set[str]:
    """Конкатенация с отсечением длинных слов (не удаляет короткие решения)."""
    return {a + b for a in left for b in right if len(a + b) <= LIMIT}


def language(expression: app.Regex) -> set[str]:
    """Вычислить язык дерева независимо от символического алгоритма."""
    if expression.kind == "empty":
        return set()
    if expression.kind == "epsilon":
        return {""}
    if expression.kind == "literal":
        return {expression.literal}
    children = [language(child) for child in expression.children]
    if expression.kind == "union":
        return set().union(*children)
    result = {""}
    if expression.kind == "concat":
        for child in children:
            result = product(result, child)
        return result
    while True:
        expanded = result | product(result, children[0])
        if expanded == result:
            return result
        result = expanded


def fixed_point(system: app.EquationSystem) -> list[set[str]]:
    """Итерации на конечной решётке языков слов длиной не более LIMIT."""
    coefficients = [[language(item) for item in row] for row in system.coefficients]
    constants = [language(item) for item in system.constants]
    current = [set() for _ in system.variables]
    while True:
        updated = [constants[i] | set().union(*(product(coefficients[i][j], current[j])
                                               for j in range(len(current))))
                   for i in range(len(current))]
        if updated == current:
            return current
        current = updated


class EquationTests(unittest.TestCase):
    def test_rule_input_and_requested_notation(self):
        system = app.read_system(ROOT / "examples" / "equations_rules.csv")
        solutions = app.solve(system)
        self.assertEqual(app.render(solutions["S"]), "2*1(a+eps)")
        self.assertEqual(app.render(solutions["A"]), "a+eps")
        self.assertEqual([language(solutions[v]) for v in system.variables], fixed_point(system))
        self.assertEqual(app.parse_regex("eps"), app.EPSILON)
        self.assertEqual(language(app.parse_regex(r"\eps")), {"eps"})
        self.assertEqual(app.parse_regex("a+b"), app.parse_regex("a|b"))
        self.assertEqual(language(app.parse_regex(r"\+")), {"+"})

    def test_rule_syntax_validation(self):
        invalid = ["S->aA", "S->aSb", "S->SS", "S->S*", "S->a|", "S->",
                   "S->a\nS->b", "SS->a", "S->a\n\n", "S->a,b", "S->a\nнеправило"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.csv"
            for text in invalid:
                with self.subTest(text=text):
                    path.write_text(text, encoding="utf-8")
                    with self.assertRaises(app.InputError):
                        app.read_system(path)
            for text in ["S->(a|b)*A|eps\nA->bS|a", "S->eps", "S->S",
                         "S->(aS|bS)|eps", r"S->\A|\eps", "S->A\nA->a"]:
                with self.subTest(text=text):
                    path.write_text(text, encoding="utf-8-sig")
                    system = app.read_system(path)
                    actual = app.solve(system)
                    self.assertEqual([language(actual[v]) for v in system.variables], fixed_point(system))

    def test_rule_cli(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.csv"
            path.write_bytes((ROOT / "examples" / "equations_rules.csv").read_bytes())
            process = subprocess.run([sys.executable, str(ROOT / "regular_equations_solver.py")],
                                     input=f"{path}\n\n\n", text=True, capture_output=True, timeout=10)
            self.assertEqual(process.returncode, 0, process.stderr)
            self.assertIn("S=2*1(a+eps)", process.stdout)
            self.assertIn("A=a+eps", process.stdout)
            with (path.parent / "equations_solution.csv").open(encoding="utf-8", newline="") as stream:
                self.assertEqual(list(csv.reader(stream)),
                                 [["переменная", "решение"], ["S", "2*1(a+eps)"], ["A", "a+eps"]])

    def test_syntax_and_roundtrip(self):
        expressions = ["a|bc*", "(a|b)*ab", "ε|∅", r"\|\*\(\)\\\ε\∅\ ",
                       "a**", " a ( b | ε ) * ", "(∅)*", "a|a", "aε"]
        for text in expressions:
            with self.subTest(text=text):
                tree = app.parse_regex(text)
                self.assertEqual(tree, app.parse_regex(app.render(tree)))
        self.assertEqual(language(app.parse_regex("a|bc*")), {"a", "b", "bc", "bcc", "bccc"})

    def test_malformed_syntax(self):
        for text in ("", " ", "()", "a|", "|a", "*a", "(a", "a)", "\\", "a||b"):
            with self.subTest(text=text), self.assertRaises(app.InputError):
                app.parse_regex(text)

    def test_scalar_cycles_and_least_solution(self):
        for coefficient, constant, expected in [
            ("a", "b", "a*b"), ("ε", "a", "a"), ("ε", "∅", "∅"),
            ("a|ε", "b", "(a|ε)*b"), ("∅", "ε", "ε"),
        ]:
            system = app.EquationSystem(("X",), ((app.parse_regex(coefficient),),),
                                        (app.parse_regex(constant),))
            self.assertEqual(language(app.solve(system)["X"]), language(app.parse_regex(expected)))

    def test_seeded_systems_against_fixed_point(self):
        rng = random.Random(23)
        choices = [app.parse_regex(text) for text in ("∅", "ε", "a", "b", "a|ε", "ab", "a*")]
        for _ in range(45):
            n = rng.randint(1, 3)
            system = app.EquationSystem(tuple(f"X{i}" for i in range(n)),
                                        tuple(tuple(rng.choice(choices) for _ in range(n)) for _ in range(n)),
                                        tuple(rng.choice(choices) for _ in range(n)))
            actual = app.solve(system)
            self.assertEqual([language(actual[v]) for v in system.variables], fixed_point(system))
            for value in actual.values():
                self.assertEqual(language(value), language(app.parse_regex(app.render(value))))

    def test_csv_errors_and_row_order(self):
        cases = ["", "переменная,X,свободный_член\n",
                 "переменная,X,X,свободный_член\nX,a,b,ε\n",
                 "переменная,X,свободный_член\nY,a,ε\n",
                 "переменная,X,свободный_член\nX,a,\n",
                 "переменная,X,свободный_член\nX,a,ε\nX,a,ε\n",
                 "переменная,X,свободный_член\nX,(a,ε\n"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.csv"
            for text in cases:
                with self.subTest(text=text):
                    path.write_text(text, encoding="utf-8")
                    with self.assertRaises(app.InputError):
                        app.read_system(path)
            path.write_text("\ufeffпеременная,X,Y,свободный_член\nY,∅,b,a\nX,a,b,ε\n", encoding="utf-8")
            system = app.read_system(path)
            self.assertEqual(system.variables, ("X", "Y"))
            self.assertEqual([language(value) for value in app.solve(system).values()], fixed_point(system))

    def test_safe_saving_concurrent_and_directory_collision(self):
        for module in (app, dfa_minimizer):
            with self.subTest(module=module.__name__), tempfile.TemporaryDirectory() as directory:
                parent = Path(directory)
                (parent / "result.csv").mkdir()
                (parent / "result_1.csv").write_text("не изменять", encoding="utf-8")
                with ThreadPoolExecutor(max_workers=6) as pool:
                    paths = list(pool.map(lambda _: module.save_rows(parent, "result", [["данные"]]), range(12)))
                self.assertEqual(len(set(paths)), 12)
                self.assertEqual((parent / "result_1.csv").read_text(encoding="utf-8"), "не изменять")
                for path in paths:
                    self.assertTrue(path.is_absolute())
                    self.assertEqual(path.read_text(encoding="utf-8"), "данные\n")

    def test_cli_and_saved_solution(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.csv"
            path.write_bytes((ROOT / "examples" / "equations.csv").read_bytes())
            for suffix in ("", "_1"):
                process = subprocess.run([sys.executable, str(ROOT / "regular_equations_solver.py")],
                                         input=f"{path.parent / 'missing.csv'}\n{path}\n\n\n",
                                         text=True, capture_output=True, timeout=10)
                self.assertEqual(process.returncode, 0, process.stderr)
                output = path.parent / f"equations_solution{suffix}.csv"
                self.assertIn(str(output), process.stdout)
                with output.open(encoding="utf-8", newline="") as stream:
                    rows = list(csv.reader(stream))
                self.assertEqual(rows[0], ["переменная", "решение"])
                expected = fixed_point(app.read_system(path))
                self.assertEqual([language(app.parse_regex(row[1])) for row in rows[1:]], expected)

    def test_io_edge_cases(self):
        for module, reader in ((app, app.read_system), (dfa_minimizer, dfa_minimizer.read_dfa)):
            with self.subTest(module=module.__name__), tempfile.TemporaryDirectory() as directory:
                parent = Path(directory)
                path = parent / "invalid.csv"
                path.write_bytes(b"\xff\xfe")
                with self.assertRaises(UnicodeError):
                    reader(path)
                self.assertIn("UTF-8", module.describe_error(UnicodeError()))
                self.assertIn("Недопустимое", module.describe_error(ValueError("embedded null byte")))
                link = parent / "result.csv"
                link.symlink_to(parent / "missing.csv")
                output = module.save_rows(parent, "result", [["данные"]])
                self.assertEqual(output.name, "result_1.csv")
                self.assertFalse((parent / "missing.csv").exists())
                with patch.object(Path, "open", side_effect=PermissionError()):
                    with self.assertRaises(PermissionError):
                        module.save_rows(parent, "blocked", [["данные"]])
                # Кодирование ломается уже после создания файла: требуется очистка.
                with self.assertRaises(UnicodeError):
                    module.save_rows(parent, "broken", [["\ud800"]])
                self.assertFalse((parent / "broken.csv").exists())

    def test_end_of_input(self):
        for name in ("dfa_minimizer.py", "regular_equations_solver.py"):
            process = subprocess.run([sys.executable, str(ROOT / name)], input="", text=True,
                                     capture_output=True, timeout=10)
            self.assertEqual(process.returncode, 0)
            self.assertIn("Ввод завершён", process.stdout)


if __name__ == "__main__":
    unittest.main()