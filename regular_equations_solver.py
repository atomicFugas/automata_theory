#!/usr/bin/env python3
"""Наименьшее решение праволинейной системы регулярных уравнений. Python 3.11+."""
from __future__ import annotations
import csv
import io
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


class InputError(ValueError):
    """Ошибка формата системы или синтаксиса регулярного выражения."""


@dataclass(frozen=True)
class Regex:
    """Неизменяемое дерево: пустой язык, пустое слово, литерал и операции."""

    kind: str
    children: tuple[Regex, ...] = ()
    literal: str = ""


EMPTY = Regex("empty")
EPSILON = Regex("epsilon")


def union(*expressions: Regex) -> Regex:
    """Объединить языки, удалив пустой язык и структурные повторы."""
    terms: dict[Regex, None] = {}
    for expression in expressions:
        if expression == EMPTY:
            continue
        for term in expression.children if expression.kind == "union" else (expression,):
            terms[term] = None
    if not terms:
        return EMPTY
    if len(terms) == 1:
        return next(iter(terms))
    return Regex("union", tuple(terms))


def concat(*expressions: Regex) -> Regex:
    """Конкатенировать без изменения порядка множителей."""
    factors: list[Regex] = []
    for expression in expressions:
        if expression == EMPTY:
            return EMPTY
        if expression == EPSILON:
            continue
        factors.extend(expression.children if expression.kind == "concat" else (expression,))
    if not factors:
        return EPSILON
    if len(factors) == 1:
        return factors[0]
    return Regex("concat", tuple(factors))


def star(expression: Regex) -> Regex:
    """Применить тождества ∅* = ε* = ε и (A*)* = A*."""
    if expression in (EMPTY, EPSILON):
        return EPSILON
    if expression.kind == "star":
        return expression
    return Regex("star", (expression,))


def render(expression: Regex, parent_priority: int = 0) -> str:
    """Напечатать однозначное выражение с минимально нужными скобками."""
    priorities = {"union": 1, "concat": 2, "star": 3,
                  "empty": 4, "epsilon": 4, "literal": 4}
    priority = priorities[expression.kind]
    if expression.kind == "empty":
        text = "∅"
    elif expression.kind == "epsilon":
        text = "eps"
    elif expression.kind == "literal":
        char = expression.literal
        text = "\\" + char if char in "|+*()\\ε∅e" or char.isspace() else char
    elif expression.kind == "star":
        text = render(expression.children[0], priority) + "*"
    else:
        separator = "+" if expression.kind == "union" else ""
        text = separator.join(render(child, priority) for child in expression.children)
    return f"({text})" if priority < parent_priority else text


class Parser:
    """Рекурсивный спуск: объединение, конкатенация, повторение, атом."""

    def __init__(self, text: str, *, rule_mode: bool = False):
        self.text = text
        self.position = 0
        self.rule_mode = rule_mode

    def peek(self) -> str:
        """Пропустить неэкранированные пробелы и вернуть следующий символ."""
        while self.position < len(self.text) and self.text[self.position].isspace():
            self.position += 1
        return self.text[self.position] if self.position < len(self.text) else ""

    def fail(self, message: str) -> None:
        """Указать позицию ошибки в исходном выражении."""
        raise InputError(f"{message} Позиция: {self.position + 1}.")

    def parse(self) -> Regex:
        """Разобрать всё выражение, не допуская лишних символов."""
        result = self.parse_union()
        if self.peek():
            self.fail("Неожиданный символ.")
        return result

    def parse_union(self) -> Regex:
        terms = [self.parse_concat()]
        while self.peek() in ("|", "+"):
            self.position += 1
            terms.append(self.parse_concat())
        return union(*terms)

    def parse_concat(self) -> Regex:
        factors: list[Regex] = []
        while self.peek() and self.peek() not in "|+)":
            factors.append(self.parse_repeat())
        if not factors:
            self.fail("Ожидалось выражение; используйте ε или ∅ для пустых значений.")
        return concat(*factors)

    def parse_repeat(self) -> Regex:
        char = self.peek()
        if self.text.startswith("eps", self.position):
            self.position += 3
            result = EPSILON
            while self.peek() == "*":
                self.position += 1
                result = star(result)
            return result
        self.position += 1
        if char == "(":
            result = self.parse_union()
            if self.peek() != ")":
                self.fail("Не найдена закрывающая скобка.")
            self.position += 1
        elif char == "\\":
            if self.position >= len(self.text):
                self.fail("После обратной косой черты нужен литерал.")
            literal = self.text[self.position]
            self.position += 1
            result = Regex("literal", literal=literal)
        elif char == "*":
            self.fail("Звёздочка должна следовать за выражением.")
        elif char == "ε":
            result = EPSILON
        elif char == "∅":
            result = EMPTY
        elif self.rule_mode and "A" <= char <= "Z":
            result = Regex("variable", literal=char)
        else:
            result = Regex("literal", literal=char)
        while self.peek() == "*":
            self.position += 1
            result = star(result)
        return result


def parse_regex(text: str) -> Regex:
    """Разобрать регулярное выражение; ε обозначает именно пустое слово."""
    return Parser(text).parse()


@dataclass(frozen=True)
class EquationSystem:
    """Система X = AX ∪ B; коэффициенты всегда стоят слева от переменных."""

    variables: tuple[str, ...]
    coefficients: tuple[tuple[Regex, ...], ...]
    constants: tuple[Regex, ...]


def linearize(expression: Regex, variables: tuple[str, ...]) -> tuple[dict[str, Regex], Regex]:
    """Преобразовать дерево правила в коэффициенты, отвергая нелинейность.

    Распределение выполняется символически; переменная допустима только
    последним множителем. Например, (a|b)A и aA|bA задают один коэффициент.
    """
    if expression.kind == "variable":
        if expression.literal not in variables:
            raise InputError(f"Нет правила для переменной «{expression.literal}».")
        return {expression.literal: EPSILON}, EMPTY
    if expression.kind in {"literal", "empty", "epsilon"}:
        return {}, expression
    if expression.kind == "star":
        coefficients, constant = linearize(expression.children[0], variables)
        if coefficients:
            raise InputError("Нельзя применять звёздочку к выражению с переменными.")
        return {}, star(constant)
    coefficients: dict[str, Regex] = {}
    constant = EMPTY if expression.kind == "union" else EPSILON
    for child in expression.children:
        right, free = linearize(child, variables)
        if expression.kind == "union":
            for name, value in right.items():
                coefficients[name] = union(coefficients.get(name, EMPTY), value)
            constant = union(constant, free)
        else:
            if coefficients and (right or free != EPSILON):
                raise InputError("Правило должно быть праволинейным: переменная только в конце слагаемого.")
            for name, value in right.items():
                coefficients[name] = concat(constant, value)
            constant = concat(constant, free)
    return coefficients, constant


def read_rules(rows: list[tuple[int, list[str]]]) -> EquationSystem:
    """Прочитать одно правило S->1A|2S в строке CSV без заголовка.
    Переменные — одиночные латинские заглавные буквы. Экранированная
    заглавная буква означает терминал. Повторная левая часть запрещена.
    """
    rules: dict[str, tuple[int, str]] = {}
    for line, row in rows:
        if len(row) != 1 or row[0].count("->") != 1:
            raise InputError(f"Строка {line}: ожидается одно правило вида S->1A|2S.")
        name, body = (part.strip() for part in row[0].split("->"))
        if len(name) != 1 or not "A" <= name <= "Z":
            raise InputError(f"Строка {line}: переменная должна быть одной буквой A–Z.")
        if name in rules:
            raise InputError(f"Строка {line}: повторное правило для «{name}»; объедините альтернативы через |.")
        rules[name] = (line, body)
    variables = tuple(rules)
    coefficients: list[tuple[Regex, ...]] = []
    constants: list[Regex] = []
    for line, body in rules.values():
        try:
            terms, constant = linearize(Parser(body, rule_mode=True).parse(), variables)
        except InputError as error:
            raise InputError(f"Строка {line}: {error}") from error
        coefficients.append(tuple(terms.get(name, EMPTY) for name in variables))
        constants.append(constant)
    return EquationSystem(variables, tuple(coefficients), tuple(constants))


def read_system(path: Path) -> EquationSystem:
    """Прочитать правила без заголовка либо прежний матричный CSV."""
    if path.suffix.lower() != ".csv":
        raise InputError("Входной файл должен иметь расширение .csv.")
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream, strict=True)
        header = next(reader, None)
        if header is None:
            raise InputError("Входной файл пуст.")
        if len(header) == 1 and "->" in header[0]:
            rows = [(reader.line_num, header)]
            rows.extend((reader.line_num, row) for row in reader)
            return read_rules(rows)
        header = [cell.strip() for cell in header]
        if len(header) < 3 or header[0] != "переменная" or header[-1] != "свободный_член":
            raise InputError("Ожидаются столбцы: переменная, имена переменных, свободный_член.")
        if any(not cell for cell in header) or len(set(header)) != len(header):
            raise InputError("Заголовки должны быть непустыми и уникальными.")
        variables = tuple(header[1:-1])
        rows: dict[str, tuple[Regex, ...]] = {}
        for row in reader:
            if len(row) != len(header):
                raise InputError(f"Строка {reader.line_num}: неверное число столбцов.")
            name = row[0].strip()
            if name not in variables or name in rows:
                raise InputError(f"Строка {reader.line_num}: неизвестная или повторная переменная «{name}».")
            expressions: list[Regex] = []
            for column, cell in enumerate(row[1:], start=1):
                try:
                    expressions.append(parse_regex(cell))
                except InputError as error:
                    raise InputError(f"Строка {reader.line_num}, столбец «{header[column]}»: {error}") from error
            rows[name] = tuple(expressions)
    if set(rows) != set(variables):
        raise InputError("Для каждой переменной требуется ровно одна строка.")
    return EquationSystem(variables, tuple(rows[v][:-1] for v in variables),
                          tuple(rows[v][-1] for v in variables))


def solve(system: EquationSystem) -> dict[str, Regex]:
    """Вычислить наименьшую неподвижную точку исключением по лемме Ардена.
    Переменные исключаются в обратном порядке входных строк, чтобы первая
    переменная (обычно стартовая) осталась последним pivot. На шаге k уравнение
    Xk = Akk Xk ∪ C заменяется на Xk = Akk* C. Это наименьшее решение
    независимо от наличия ε в Akk. Обратный проход восстанавливает переменные.
    """
    n = len(system.variables)
    if (not n or len(set(system.variables)) != n
            or any(not v.strip() for v in system.variables)
            or len(system.coefficients) != n or len(system.constants) != n
            or any(len(row) != n for row in system.coefficients)):
        raise InputError("Некорректные имена или размеры системы уравнений.")
    # Keep the last variable in the elimination order as the final pivot.
    # For rule files this tends to leave the start variable (commonly the
    # first rule) until last, avoiding unnecessarily expanded expressions.
    order = tuple(reversed(range(n)))
    variables = tuple(system.variables[index] for index in order)
    matrix = [[system.coefficients[i][j] for j in order] for i in order]
    constants = [system.constants[i] for i in order]
    for k in range(n):
        closure = star(matrix[k][k])
        constants[k] = concat(closure, constants[k])
        for j in range(k + 1, n):
            matrix[k][j] = concat(closure, matrix[k][j])
        matrix[k][k] = EMPTY
        for i in range(k + 1, n):
            factor = matrix[i][k]
            if factor != EMPTY:
                constants[i] = union(constants[i], concat(factor, constants[k]))
                for j in range(k + 1, n):
                    matrix[i][j] = union(matrix[i][j], concat(factor, matrix[k][j]))
            matrix[i][k] = EMPTY
    solutions = [EMPTY] * n
    for i in range(n - 1, -1, -1):
        solutions[i] = union(constants[i], *(concat(matrix[i][j], solutions[j])
                                             for j in range(i + 1, n)))
    by_variable = dict(zip(variables, solutions))
    return {name: by_variable[name] for name in system.variables}


def system_rows(system: EquationSystem) -> list[list[str]]:
    """Сформировать таблицу нормализованных входных выражений."""
    return [["переменная", *system.variables, "свободный_член"]] + [
        [name, *(render(item) for item in system.coefficients[i]), render(system.constants[i])]
        for i, name in enumerate(system.variables)
    ]


def equation_rows(system: EquationSystem) -> list[list[str]]:
    """Показать систему правилами, а не матрицей коэффициентов."""
    rows = [["правило"]]
    for i, name in enumerate(system.variables):
        terms = []
        for j, variable in enumerate(system.variables):
            value = system.coefficients[i][j]
            if value != EMPTY:
                terms.append(("" if value == EPSILON else render(value, 2)) + variable)
        if system.constants[i] != EMPTY:
            terms.append(render(system.constants[i]))
        rows.append([name + "->" + ("|".join(terms) if terms else "∅")])
    return rows


def format_table(rows: Sequence[Sequence[str]]) -> str:
    """Сформировать русскоязычную таблицу, экранируя переводы строк."""
    visible = [[cell.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")
                for cell in row] for row in rows]
    widths = [max(len(row[i]) for row in visible) for i in range(len(visible[0]))]
    border = "+" + "+".join("-" * (width + 2) for width in widths) + "+"
    lines = [border]
    for index, row in enumerate(visible):
        lines.append("| " + " | ".join(cell.ljust(width) for cell, width in zip(row, widths)) + " |")
        if index == 0:
            lines.append(border)
    return "\n".join([*lines, border])


def save_rows(directory: Path, stem: str, rows: Sequence[Sequence[str]]) -> Path:
    """Сохранить CSV исключительно в новый файл, безопасно при коллизиях."""
    buffer = io.StringIO(newline="")
    csv.writer(buffer).writerows(rows)
    payload = buffer.getvalue()
    number = 0
    while True:
        suffix = f"_{number}" if number else ""
        path = directory / f"{stem}{suffix}.csv"
        try:
            stream = path.open("x", encoding="utf-8", newline="")
        except FileExistsError:
            number += 1
            continue
        try:
            with stream:
                stream.write(payload)
        except BaseException:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass  # Сохраняем исходную ошибку записи, даже если очистка не удалась.
            raise
        return path.absolute()


def describe_error(error: Exception) -> str:
    """Перевести ошибки среды, не выводя англоязычных сообщений ОС."""
    if isinstance(error, UnicodeError):
        return "Файл должен быть сохранён в кодировке UTF-8."
    if isinstance(error, FileNotFoundError):
        return "Файл или каталог не найден."
    if isinstance(error, PermissionError):
        return "Недостаточно прав доступа к файлу или каталогу."
    if isinstance(error, IsADirectoryError):
        return "Вместо файла указан каталог."
    if isinstance(error, OSError):
        return f"Системная ошибка файловой операции (код {error.errno})."
    if isinstance(error, csv.Error):
        return "Некорректный CSV: проверьте кавычки, разделители и размер полей."
    if isinstance(error, RecursionError):
        return "Выражение слишком глубоко вложено для обработки."
    if isinstance(error, MemoryError):
        return "Недостаточно памяти: система или результирующие выражения слишком велики."
    if isinstance(error, InputError):
        return str(error)
    return "Недопустимое значение: проверьте путь и содержимое входного файла."


def main() -> int:
    """Интерактивно загрузить, решить и сохранить систему."""
    print("Решение системы регулярных уравнений")
    print("Вход: одно правило в строке, например S->1A|2S и A->a|eps; без заголовка.")
    print("Вычисляется наименьшее решение по включению языков; циклы допустимы.")
    try:
        while True:
            try:
                raw = input("Введите путь к входному файлу .csv: ").strip()
                if not raw:
                    raise InputError("Путь не может быть пустым.")
                path = Path(raw).expanduser().absolute()
                system = read_system(path)
                break
            except (ValueError, OSError, csv.Error, RecursionError) as error:
                print(f"Ошибка чтения: {describe_error(error)}")
        print("\nИсходная система (правила нормализованы):")
        print(format_table(equation_rows(system)))
        input("\nНажмите Enter для решения: ")
        solutions = solve(system)
        rows = [["переменная", "решение"], *[[name, render(value)] for name, value in solutions.items()]]
        print("\nНаименьшее решение:")
        for name, value in solutions.items():
            print(f"{name}={render(value)}")
        output = save_rows(path.parent, "equations_solution", rows)
        print(f"\nРезультат успешно сохранён: {output}")
        input("Нажмите Enter для выхода: ")
        return 0
    except EOFError:
        print("\nВвод завершён. Программа закрыта.")
        return 0
    except KeyboardInterrupt:
        print("\nОперация прервана пользователем.")
        return 130
    except (ValueError, OSError, csv.Error, RecursionError, MemoryError) as error:
        print(f"Ошибка: {describe_error(error)}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())