#!/usr/bin/env python3
"""Интерактивная минимизация полного ДКА алгоритмом Мура. Python 3.11+."""

from __future__ import annotations

import csv
import io
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


class InputError(ValueError):
    """Ошибка структуры или содержимого входного файла."""


class MetadataRequired(InputError):
    """Таблица содержит переходы, но начальное и допускающие состояния заданы отдельно."""


@dataclass(frozen=True)
class DFA:
    """Полный ДКА; индексы переходов соответствуют порядку алфавита."""

    states: tuple[str, ...]
    alphabet: tuple[str, ...]
    initial: str
    accepting: frozenset[str]
    transitions: dict[str, tuple[str, ...]]


@dataclass(frozen=True)
class MinimizationResult:
    """Минимальный автомат, классы эквивалентности и удалённые состояния."""

    automaton: DFA
    groups: dict[str, tuple[str, ...]]
    unreachable: tuple[str, ...]
    refinements: int


def normalize_state(text: str) -> str:
    """Канонизировать подмножество НКА или вложенный класс состояний ДКА.

    Краткая запись {AD} означает {A,D}. Для многосимвольных имён используйте
    разделитель запятую: {q0,q1}. Вложенность сохраняет классы эквивалентности,
    а не превращает их в объединение исходных состояний НКА.
    """
    text = text.strip()
    if not text:
        raise InputError("Обозначение состояния не может быть пустым.")
    if not any(char in text for char in "{},"):
        return text  # Совместимость с обычными именами состояний ДКА.
    if not (text.startswith("{") and text.endswith("}")):
        raise InputError(f"Некорректное множество состояний «{text}».")
    body = text[1:-1].strip()
    if not body:
        return "{}"  # Пустое подмножество — допустимое состояние-ловушка.
    depth = 0
    start = 0
    parts: list[str] = []
    for index, char in enumerate(body):
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth < 0:
                raise InputError(f"Нарушен баланс скобок в состоянии «{text}».")
        elif char == "," and depth == 0:
            parts.append(body[start:index].strip())
            start = index + 1
    if depth:
        raise InputError(f"Нарушен баланс скобок в состоянии «{text}».")
    parts.append(body[start:].strip())
    if any(not part for part in parts):
        raise InputError(f"Пустой элемент в состоянии «{text}».")
    nested = any("{" in part or "}" in part for part in parts)
    if nested:
        if not all(part.startswith("{") and part.endswith("}") for part in parts):
            raise InputError("Нельзя смешивать элементы НКА и множества ДКА в одном классе.")
        members = [normalize_state(part) for part in parts]
    else:
        # Краткая форма предназначена для однобуквенных исходных состояний.
        members = list(parts[0]) if len(parts) == 1 and parts[0].isalpha() else parts
        if any(any(char.isspace() for char in member) for member in members):
            raise InputError("Внутри имени элемента множества не должно быть пробелов.")
    if len(set(members)) != len(members):
        raise InputError(f"Повторяющийся элемент в состоянии «{text}».")
    return "{" + ",".join(sorted(members)) + "}"


def class_name(members: Sequence[str]) -> str:
    """Сохранить подмножества НКА как отдельные элементы класса эквивалентности."""
    return members[0] if len(members) == 1 else "{" + ",".join(sorted(members)) + "}"


def validate_dfa(dfa: DFA) -> None:
    """Проверить инварианты, включая вызовы алгоритма вне CLI."""
    states = set(dfa.states)
    if not states or len(states) != len(dfa.states):
        raise InputError("Состояния должны быть непустым списком без повторений.")
    if any(not name.strip() for name in dfa.states + dfa.alphabet):
        raise InputError("Имена состояний и символов не могут быть пустыми.")
    if len(set(dfa.alphabet)) != len(dfa.alphabet):
        raise InputError("Символы алфавита не должны повторяться.")
    if dfa.initial not in states or not dfa.accepting <= states:
        raise InputError("Начальное или допускающее состояние отсутствует в автомате.")
    if set(dfa.transitions) != states:
        raise InputError("Таблица переходов должна содержать все состояния и только их.")
    for state, targets in dfa.transitions.items():
        if len(targets) != len(dfa.alphabet) or not set(targets) <= states:
            raise InputError(f"Некорректные переходы состояния «{state}».")


def read_dfa(path: Path, *, initial: str | None = None,
             accepting: Sequence[str] | None = None) -> DFA:
    """Прочитать CSV с флагами или таблицу переходов с переданной разметкой."""
    if path.suffix.lower() != ".csv":
        raise InputError("Входной файл должен иметь расширение .csv.")
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream, strict=True)
        header = next(reader, None)
        if header is None:
            raise InputError("Входной файл пуст.")
        header = [cell.strip() for cell in header]
        has_flags = header[:3] == ["состояние", "начальное", "допускающее"]
        if not has_flags and (not header or header[0] != "состояние"):
            raise InputError("Ожидаются столбцы состояния и символов алфавита.")
        alphabet = header[3:] if has_flags else header[1:]
        if not alphabet:
            raise InputError("После столбца состояния укажите символы алфавита.")
        if len(set(header)) != len(header) or any(not cell for cell in header):
            raise InputError("Заголовки должны быть непустыми и уникальными.")
        if not has_flags and (initial is None or accepting is None):
            raise MetadataRequired(
                "Для таблицы только с переходами укажите начальное и допускающие состояния.")
        states: list[str] = []
        initial_rows: list[str] = []
        accepting_rows: set[str] = set()
        transitions: dict[str, tuple[str, ...]] = {}
        for row in reader:
            if len(row) != len(header):
                raise InputError(f"Строка {reader.line_num}: неверное число столбцов.")
            if has_flags:
                state, start, final, *targets = [cell.strip() for cell in row]
            else:
                state, *targets = [cell.strip() for cell in row]
                start = final = "0"
            state = normalize_state(state)
            targets = [normalize_state(target) for target in targets]
            if not state or state in transitions:
                raise InputError(f"Строка {reader.line_num}: пустое или повторяющееся состояние.")
            if start not in {"0", "1"} or final not in {"0", "1"}:
                raise InputError(f"Строка {reader.line_num}: флаги должны быть 0 или 1.")
            states.append(state)
            transitions[state] = tuple(targets)
            if has_flags and start == "1":
                initial_rows.append(state)
            if has_flags and final == "1":
                accepting_rows.add(state)
    if has_flags:
        if len(initial_rows) != 1:
            raise InputError("Необходимо ровно одно начальное состояние.")
        start_state = initial_rows[0]
        final_states = frozenset(accepting_rows)
    else:
        start_state = normalize_state(initial)
        final_states = frozenset(normalize_state(state) for state in accepting)
    dfa = DFA(tuple(states), tuple(alphabet), start_state, final_states, transitions)
    validate_dfa(dfa)
    return dfa


def minimize(dfa: DFA) -> MinimizationResult:
    """Удалить недостижимые состояния и уточнять классы до неподвижной точки."""
    validate_dfa(dfa)
    reachable = {dfa.initial}
    queue = deque([dfa.initial])
    while queue:
        for target in dfa.transitions[queue.popleft()]:
            if target not in reachable:
                reachable.add(target)
                queue.append(target)
    ordered = [state for state in dfa.states if state in reachable]
    groups = [group for group in (
        [s for s in ordered if s not in dfa.accepting],
        [s for s in ordered if s in dfa.accepting],
    ) if group]
    refinements = 0
    while True:
        owner = {state: index for index, group in enumerate(groups) for state in group}
        refined: list[list[str]] = []
        for group in groups:
            buckets: dict[tuple[int, ...], list[str]] = {}
            for state in group:
                signature = tuple(owner[target] for target in dfa.transitions[state])
                buckets.setdefault(signature, []).append(state)
            refined.extend(buckets.values())
        if len(refined) == len(groups):
            break
        refinements += 1
        groups = refined

    # Обход в ширину задаёт стабильный порядок строк результата.
    owner = {state: index for index, group in enumerate(groups) for state in group}
    first = owner[dfa.initial]
    order = [first]
    subset_mode = all(state.startswith("{") and state.endswith("}") for state in dfa.states)
    names = {first: class_name(groups[first]) if subset_mode else "М0"}
    for index in order:
        for target in dfa.transitions[groups[index][0]]:
            target_index = owner[target]
            if target_index not in names:
                names[target_index] = (class_name(groups[target_index]) if subset_mode
                                       else f"М{len(names)}")
                order.append(target_index)
    result = DFA(
        tuple(names[index] for index in order), dfa.alphabet, names[first],
        frozenset(names[index] for index in order if groups[index][0] in dfa.accepting),
        {names[index]: tuple(names[owner[t]] for t in dfa.transitions[groups[index][0]])
         for index in order},
    )
    return MinimizationResult(
        result, {names[index]: tuple(groups[index]) for index in order},
        tuple(s for s in dfa.states if s not in reachable), refinements,
    )


def dfa_rows(dfa: DFA) -> list[list[str]]:
    """Представить автомат в едином формате для экрана и CSV."""
    return [["состояние", "начальное", "допускающее", *dfa.alphabet]] + [
        [s, str(int(s == dfa.initial)), str(int(s in dfa.accepting)), *dfa.transitions[s]]
        for s in dfa.states
    ]


def format_table(rows: Sequence[Sequence[str]]) -> str:
    """Сформировать таблицу; управляющие символы показывать явно."""
    visible = [[cell.replace("\\", "\\\\").replace("\n", "\\n").replace("\r", "\\r")
                .replace("\t", "\\t") for cell in row] for row in rows]
    widths = [max(len(row[i]) for row in visible) for i in range(len(visible[0]))]
    border = "+" + "+".join("-" * (width + 2) for width in widths) + "+"
    lines = [border]
    for index, row in enumerate(visible):
        lines.append("| " + " | ".join(cell.ljust(width) for cell, width in zip(row, widths)) + " |")
        if index == 0:
            lines.append(border)
    return "\n".join([*lines, border])


def save_rows(directory: Path, stem: str, rows: Sequence[Sequence[str]]) -> Path:
    """Создать новый CSV без перезаписи; режим x защищает от гонки имён."""
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
            # Не оставлять незавершённый результат при ошибке записи или прерывании.
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass  # Ошибка очистки не должна скрыть исходную причину сбоя.
            raise
        return path.absolute()


def main() -> int:
    """Обработать интерактивный сеанс и вернуть код завершения."""
    print("Минимизация детерминированного конечного автомата")
    try:
        while True:
            try:
                raw = input("Введите путь к входному файлу .csv: ").strip()
                if not raw:
                    raise InputError("Путь не может быть пустым.")
                path = Path(raw).expanduser().absolute()
                try:
                    dfa = read_dfa(path)
                except MetadataRequired:
                    start_state = input("Введите начальное состояние: ").strip()
                    final_states = input(
                        "Введите допускающие состояния через запятую (или оставьте пустым): "
                    ).strip()
                    dfa = read_dfa(
                        path,
                        initial=start_state,
                        accepting=[state.strip() for state in final_states.split(",") if state.strip()],
                    )
                break
            except (ValueError, OSError, csv.Error) as error:
                print(f"Ошибка чтения: {describe_error(error)}")
        print("\nИсходный автомат:")
        print(format_table(dfa_rows(dfa)))
        input("\nНажмите Enter для минимизации: ")
        result = minimize(dfa)
        print("\nМинимальный автомат:")
        print(format_table(dfa_rows(result.automaton)))
        # print(f"Число уточнений разбиения: {result.refinements}")
        # print("Удалённые недостижимые состояния: " + (", ".join(result.unreachable) or "нет"))
        print("Состав новых состояний:")
        for name, members in result.groups.items():
            print(f"  {name}: {', '.join(members)}")
        output = save_rows(path.parent, "dfa_minimized", dfa_rows(result.automaton))
        print(f"\nРезультат успешно сохранён: {output}")
        input("Нажмите Enter для выхода: ")
        return 0
    except EOFError:
        print("\nВвод завершён. Программа закрыта.")
        return 0
    except KeyboardInterrupt:
        print("\nОперация прервана пользователем.")
        return 130
    except (ValueError, OSError, csv.Error, MemoryError, RecursionError) as error:
        print(f"Ошибка: {describe_error(error)}", file=sys.stderr)
        return 1


def describe_error(error: Exception) -> str:
    """Не выводить англоязычные системные сообщения в интерфейс."""
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
    if isinstance(error, MemoryError):
        return "Недостаточно памяти для обработки автомата."
    if isinstance(error, RecursionError):
        return "Слишком большая вложенность множеств состояний."
    if isinstance(error, InputError):
        return str(error)
    return "Недопустимое значение: проверьте путь и содержимое входного файла."


if __name__ == "__main__":
    sys.exit(main())
