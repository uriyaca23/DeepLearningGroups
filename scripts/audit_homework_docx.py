#!/usr/bin/env python3
"""Fail-closed OOXML audit for the HW4 Word document.

This script is intentionally standard-library only.  It checks the mechanical
rules in style/homework_style_contract.json without opening or modifying Word.
It does not replace the required full-page Microsoft Word visual review.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import posixpath
import re
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET


W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
M_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"
R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
WP_NS = "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
A_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"

NS = {
    "w": W_NS,
    "m": M_NS,
    "r": R_NS,
    "rel": REL_NS,
    "wp": WP_NS,
    "a": A_NS,
}


def q(namespace: str, local: str) -> str:
    return f"{{{namespace}}}{local}"


def wattr(element: ET.Element | None, local: str) -> str | None:
    if element is None:
        return None
    return element.get(q(W_NS, local))


def mattr(element: ET.Element | None, local: str) -> str | None:
    if element is None:
        return None
    return element.get(q(M_NS, local))


def bool_value(element: ET.Element | None) -> bool | None:
    if element is None:
        return None
    value = wattr(element, "val")
    if value is None:
        return True
    return value.lower() not in {"0", "false", "off", "no"}


def clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def paragraph_text(paragraph: ET.Element) -> str:
    return clean_text("".join(node.text or "" for node in paragraph.findall(".//w:t", NS)))


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest().upper()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def nested_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = nested_merge(result[key], value)
        else:
            result[key] = value
    return result


def ppr_values(ppr: ET.Element | None) -> dict[str, Any]:
    if ppr is None:
        return {}
    spacing = ppr.find("w:spacing", NS)
    outline = ppr.find("w:outlineLvl", NS)
    jc = ppr.find("w:jc", NS)
    return {
        key: value
        for key, value in {
            "bidi": bool_value(ppr.find("w:bidi", NS)),
            "keep_next": bool_value(ppr.find("w:keepNext", NS)),
            "keep_lines": bool_value(ppr.find("w:keepLines", NS)),
            "spacing_before_twips": (
                int(wattr(spacing, "before")) if wattr(spacing, "before") is not None else None
            ),
            "spacing_after_twips": (
                int(wattr(spacing, "after")) if wattr(spacing, "after") is not None else None
            ),
            "outline_level": (
                int(wattr(outline, "val")) if wattr(outline, "val") is not None else None
            ),
            "alignment": wattr(jc, "val"),
        }.items()
        if value is not None
    }


def rpr_values(rpr: ET.Element | None) -> dict[str, Any]:
    if rpr is None:
        return {}
    fonts = rpr.find("w:rFonts", NS)
    size = rpr.find("w:sz", NS)
    size_cs = rpr.find("w:szCs", NS)
    color = rpr.find("w:color", NS)
    language = rpr.find("w:lang", NS)
    font_values: dict[str, str] = {}
    if fonts is not None:
        for key in ("ascii", "eastAsia", "hAnsi", "cs"):
            value = wattr(fonts, key)
            if value is not None:
                font_values[key] = value
    result: dict[str, Any] = {
        "fonts": font_values,
        "size_half_points": (
            int(wattr(size, "val")) if wattr(size, "val") is not None else None
        ),
        "complex_script_size_half_points": (
            int(wattr(size_cs, "val")) if wattr(size_cs, "val") is not None else None
        ),
        "color": wattr(color, "val"),
        "bidi_language": wattr(language, "bidi"),
        "bold": bool_value(rpr.find("w:b", NS)),
    }
    return {key: value for key, value in result.items() if value is not None and value != {}}


@dataclass
class Style:
    style_id: str
    name: str
    based_on: str | None
    ppr: dict[str, Any]
    rpr: dict[str, Any]


@dataclass
class Paragraph:
    index: int
    element: ET.Element
    style_id: str
    text: str
    ppr: dict[str, Any]
    drawings: list[ET.Element]
    math: list[ET.Element]
    math_paragraphs: list[ET.Element]


class Auditor:
    def __init__(self, docx_path: Path, contract_path: Path) -> None:
        self.docx_path = docx_path
        self.contract_path = contract_path
        self.contract = json.loads(contract_path.read_text(encoding="utf-8"))
        self.errors: list[str] = []
        self.notes: list[str] = []
        self.styles: dict[str, Style] = {}
        self.style_name_by_id: dict[str, str] = {}
        self.default_ppr: dict[str, Any] = {}
        self.default_rpr: dict[str, Any] = {}
        self.relationships: dict[str, str] = {}
        self.paragraphs: list[Paragraph] = []

    def error(self, message: str) -> None:
        self.errors.append(message)

    def load_styles(self, package: zipfile.ZipFile) -> None:
        root = ET.fromstring(package.read("word/styles.xml"))
        self.default_ppr = ppr_values(
            root.find("w:docDefaults/w:pPrDefault/w:pPr", NS)
        )
        self.default_rpr = rpr_values(
            root.find("w:docDefaults/w:rPrDefault/w:rPr", NS)
        )
        for element in root.findall("w:style", NS):
            style_id = wattr(element, "styleId")
            if not style_id:
                continue
            name = wattr(element.find("w:name", NS), "val") or style_id
            based_on = wattr(element.find("w:basedOn", NS), "val")
            style = Style(
                style_id=style_id,
                name=name,
                based_on=based_on,
                ppr=ppr_values(element.find("w:pPr", NS)),
                rpr=rpr_values(element.find("w:rPr", NS)),
            )
            self.styles[style_id] = style
            self.style_name_by_id[style_id] = name

    def effective_style(self, style_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
        chain: list[Style] = []
        seen: set[str] = set()
        current = style_id
        while current in self.styles and current not in seen:
            seen.add(current)
            style = self.styles[current]
            chain.append(style)
            current = style.based_on or ""
        ppr = dict(self.default_ppr)
        rpr = dict(self.default_rpr)
        for style in reversed(chain):
            ppr = nested_merge(ppr, style.ppr)
            rpr = nested_merge(rpr, style.rpr)
        return ppr, rpr

    def load_relationships(self, package: zipfile.ZipFile) -> None:
        root = ET.fromstring(package.read("word/_rels/document.xml.rels"))
        for relationship in root.findall("rel:Relationship", NS):
            rel_id = relationship.get("Id")
            target = relationship.get("Target")
            if rel_id and target:
                self.relationships[rel_id] = target

    def load_document(self, package: zipfile.ZipFile) -> ET.Element:
        root = ET.fromstring(package.read("word/document.xml"))
        body = root.find("w:body", NS)
        if body is None:
            self.error("word/document.xml has no w:body.")
            return root
        for index, element in enumerate(body.findall("w:p", NS)):
            ppr_element = element.find("w:pPr", NS)
            style_id = wattr(
                ppr_element.find("w:pStyle", NS) if ppr_element is not None else None,
                "val",
            ) or "Normal"
            self.paragraphs.append(
                Paragraph(
                    index=index,
                    element=element,
                    style_id=style_id,
                    text=paragraph_text(element),
                    ppr=ppr_values(ppr_element),
                    drawings=element.findall(".//w:drawing", NS),
                    math=element.findall(".//m:oMath", NS),
                    math_paragraphs=element.findall(".//m:oMathPara", NS),
                )
            )
        return root

    def check_geometry(self, document_root: ET.Element) -> None:
        expected = self.contract["page_geometry"]
        sections = document_root.findall(".//w:sectPr", NS)
        if not sections:
            self.error("No section properties were found.")
            return
        for section_number, section in enumerate(sections, start=1):
            page_size = section.find("w:pgSz", NS)
            margins = section.find("w:pgMar", NS)
            actual = {
                "width_twips": int(wattr(page_size, "w") or -1),
                "height_twips": int(wattr(page_size, "h") or -1),
                "top_margin_twips": int(wattr(margins, "top") or -1),
                "bottom_margin_twips": int(wattr(margins, "bottom") or -1),
                "left_margin_twips": int(wattr(margins, "left") or -1),
                "right_margin_twips": int(wattr(margins, "right") or -1),
                "header_distance_twips": int(wattr(margins, "header") or -1),
                "footer_distance_twips": int(wattr(margins, "footer") or -1),
            }
            for key, required in expected.items():
                if actual[key] != required:
                    self.error(
                        f"Section {section_number}: {key} is {actual[key]}, required {required}."
                    )

    def compare_properties(
        self,
        label: str,
        actual: dict[str, Any],
        expected: dict[str, Any],
    ) -> None:
        for key, required in expected.items():
            value = actual.get(key)
            # In OOXML, an absent on/off property such as w:b means false.
            if required is False and value is None:
                continue
            if isinstance(required, dict):
                if not isinstance(value, dict):
                    self.error(f"{label}: {key} is missing; required {required!r}.")
                    continue
                self.compare_properties(f"{label}.{key}", value, required)
            elif value != required:
                self.error(f"{label}: {key} is {value!r}; required {required!r}.")

    def check_styles(self) -> None:
        for label, rule in self.contract["styles"].items():
            style_id = rule["style_id"]
            style = self.styles.get(style_id)
            if style is None:
                self.error(f"Required style {label!r} ({style_id}) is missing.")
                continue
            actual_based_on = style.based_on
            if actual_based_on != rule["based_on"]:
                self.error(
                    f"Style {label}: basedOn is {actual_based_on!r}; "
                    f"required {rule['based_on']!r}."
                )
            effective_ppr, effective_rpr = self.effective_style(style_id)
            self.compare_properties(
                f"Style {label} paragraph", effective_ppr, rule["paragraph"]
            )
            self.compare_properties(
                f"Style {label} run", effective_rpr, rule["run"]
            )

    def paragraph_effective_properties(
        self, paragraph: Paragraph
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        style_ppr, style_rpr = self.effective_style(paragraph.style_id)
        return nested_merge(style_ppr, paragraph.ppr), style_rpr

    def check_forbidden_styles(self) -> None:
        forbidden = {
            value.casefold()
            for value in self.contract["semantic_structure"][
                "forbidden_used_style_names"
            ]
        }
        for paragraph in self.paragraphs:
            name = self.style_name_by_id.get(paragraph.style_id, paragraph.style_id)
            if name.casefold() in forbidden:
                self.error(
                    f"Paragraph {paragraph.index + 1} uses forbidden style "
                    f"{name!r}: {paragraph.text[:100]!r}."
                )

    def check_semantic_structure(self) -> set[int]:
        rules = self.contract["semantic_structure"]
        question_pattern = re.compile(rules["question_heading_regex"])
        subsection_pattern = re.compile(rules["subsection_heading_regex"])
        prompt_indices: set[int] = set()
        question_count = 0
        title_matches = [
            paragraph
            for paragraph in self.paragraphs
            if paragraph.text == rules["required_title_text"]
        ]
        if len(title_matches) != 1:
            self.error(
                f"Expected exactly one title {rules['required_title_text']!r}; "
                f"found {len(title_matches)}."
            )
        elif title_matches[0].style_id != rules["title_style_id"]:
            self.error(
                f"Document title uses {title_matches[0].style_id}; "
                f"required {rules['title_style_id']}."
            )

        for paragraph in self.paragraphs:
            if (
                paragraph.style_id == rules["question_style_id"]
                and not question_pattern.fullmatch(paragraph.text)
            ):
                self.error(
                    f"Paragraph {paragraph.index + 1} uses Heading 1 but is not a "
                    f"valid question title: {paragraph.text[:100]!r}."
                )
            if question_pattern.fullmatch(paragraph.text):
                question_count += 1
                if paragraph.style_id != rules["question_style_id"]:
                    self.error(
                        f"Paragraph {paragraph.index + 1} {paragraph.text!r} uses "
                        f"{paragraph.style_id}; required {rules['question_style_id']}."
                    )
                next_paragraph = next(
                    (
                        candidate
                        for candidate in self.paragraphs[paragraph.index + 1 :]
                        if candidate.text
                        or candidate.drawings
                        or candidate.math
                    ),
                    None,
                )
                if next_paragraph is None:
                    self.error(
                        f"{paragraph.text}: no prompt excerpt follows the question heading."
                    )
                elif not next_paragraph.drawings:
                    style_name = self.style_name_by_id.get(
                        next_paragraph.style_id, next_paragraph.style_id
                    )
                    self.error(
                        f"{paragraph.text}: the first content after Heading 1 is not a "
                        f"source-excerpt image (paragraph {next_paragraph.index + 1}, "
                        f"style {style_name!r})."
                    )
                else:
                    first_prompt_index = next_paragraph.index
                    for candidate in self.paragraphs[first_prompt_index:]:
                        if candidate.index != first_prompt_index:
                            if candidate.text or candidate.math:
                                break
                            if not candidate.drawings:
                                continue
                            if (
                                candidate.style_id
                                != self.contract["prompt_excerpts"][
                                    "paragraph_style_id"
                                ]
                            ):
                                break
                        if not candidate.drawings:
                            break
                        prompt_indices.add(candidate.index)
                        self.check_prompt_paragraph(paragraph.text, candidate)

            if subsection_pattern.match(paragraph.text):
                if paragraph.style_id != rules["subsection_style_id"]:
                    self.error(
                        f"Paragraph {paragraph.index + 1} {paragraph.text[:80]!r} "
                        f"uses {paragraph.style_id}; required "
                        f"{rules['subsection_style_id']}."
                    )

        if question_count == 0:
            self.error("No question headings matching the contract were found.")
        return prompt_indices

    def check_prompt_paragraph(self, question: str, paragraph: Paragraph) -> None:
        rule = self.contract["prompt_excerpts"]
        if paragraph.style_id != rule["paragraph_style_id"]:
            self.error(
                f"{question}: prompt paragraph uses {paragraph.style_id}; "
                f"required {rule['paragraph_style_id']}."
            )
        expected_ppr = {
            "alignment": rule["paragraph_alignment"],
            "keep_lines": rule["keep_lines"],
            "spacing_before_twips": rule["spacing_before_twips"],
            "spacing_after_twips": rule["spacing_after_twips"],
        }
        self.compare_properties(
            f"{question} prompt paragraph", paragraph.ppr, expected_ppr
        )
        if len(paragraph.drawings) < rule["minimum_drawings"]:
            self.error(f"{question}: prompt paragraph contains no source drawing.")
        for drawing in paragraph.drawings:
            extent = drawing.find(".//wp:extent", NS)
            if extent is None or extent.get("cx") is None:
                self.error(f"{question}: prompt drawing has no measurable width.")
                continue
            width_inches = int(extent.get("cx", "0")) / 914400
            if not (
                rule["minimum_width_inches"]
                <= width_inches
                <= rule["maximum_width_inches"]
            ):
                self.error(
                    f"{question}: prompt drawing width is {width_inches:.3f} inches; "
                    f"required {rule['minimum_width_inches']:.2f}-"
                    f"{rule['maximum_width_inches']:.2f}."
                )

    def check_hebrew_paragraphs(self) -> None:
        rule = self.contract["hebrew_content"]
        checked_styles = set(rule["checked_style_ids"])
        forbidden_fonts = {font.casefold() for font in rule["forbidden_fonts"]}
        deviations = rule.get("accepted_legacy_paragraph_deviations", {})
        font_deviations = set(deviations.get("font_text_sha256", []))
        direction_deviations = set(
            deviations.get("paragraph_direction_text_sha256", [])
        )
        hebrew_pattern = re.compile(r"[\u0590-\u05FF]")

        for paragraph in self.paragraphs:
            if paragraph.style_id not in checked_styles:
                continue
            if not hebrew_pattern.search(paragraph.text):
                continue
            text_digest = sha256_bytes(paragraph.text.encode("utf-8"))
            effective_ppr, style_rpr = self.paragraph_effective_properties(paragraph)
            if (
                effective_ppr.get("bidi") is not rule["required_bidi"]
                and text_digest not in direction_deviations
            ):
                self.error(
                    f"Paragraph {paragraph.index + 1} is Hebrew but not effectively RTL: "
                    f"{paragraph.text[:100]!r}."
                )
            alignment = effective_ppr.get("alignment")
            if alignment in {"left", "center"}:
                self.error(
                    f"Paragraph {paragraph.index + 1} is Hebrew but has incompatible "
                    f"alignment {alignment!r}: {paragraph.text[:100]!r}."
                )

            for run in paragraph.element.findall("w:r", NS):
                run_text = "".join(node.text or "" for node in run.findall(".//w:t", NS))
                if not hebrew_pattern.search(run_text):
                    continue
                effective_rpr = nested_merge(
                    style_rpr, rpr_values(run.find("w:rPr", NS))
                )
                if text_digest in font_deviations:
                    continue
                fonts = effective_rpr.get("fonts", {})
                actual_fonts = {
                    value for key, value in fonts.items() if key in {"ascii", "hAnsi", "cs"}
                }
                if actual_fonts != {rule["required_font"]}:
                    self.error(
                        f"Paragraph {paragraph.index + 1} Hebrew run does not resolve "
                        f"exclusively to {rule['required_font']!r}: {run_text[:80]!r}; "
                        f"fonts={sorted(actual_fonts)!r}."
                    )
                bad = sorted(
                    font
                    for font in actual_fonts
                    if font.casefold() in forbidden_fonts
                )
                if bad:
                    self.error(
                        f"Paragraph {paragraph.index + 1} Hebrew run uses forbidden "
                        f"font(s) {bad!r}: {run_text[:80]!r}."
                    )

    def check_equations(self) -> None:
        if not self.contract["display_equations"]["require_centered"]:
            return
        for paragraph in self.paragraphs:
            if not paragraph.math:
                continue
            visible_text = paragraph.text
            if visible_text:
                continue
            paragraph_centered = paragraph.ppr.get("alignment") == "center"
            math_centered = any(
                mattr(math_para.find("m:oMathParaPr/m:jc", NS), "val") == "center"
                for math_para in paragraph.math_paragraphs
            )
            if not paragraph_centered and not math_centered:
                self.error(
                    f"Display equation paragraph {paragraph.index + 1} is not centered."
                )

    def drawing_target(self, drawing: ET.Element) -> str | None:
        blip = drawing.find(".//a:blip", NS)
        if blip is None:
            return None
        rel_id = blip.get(q(R_NS, "embed"))
        if not rel_id:
            return None
        target = self.relationships.get(rel_id)
        if not target:
            return None
        return posixpath.normpath(posixpath.join("word", target))

    def check_media(
        self, package: zipfile.ZipFile, prompt_indices: set[int]
    ) -> None:
        allowed = {
            item["sha256"].upper()
            for item in self.contract["allowed_non_prompt_media_sha256"]
        }
        for paragraph in self.paragraphs:
            for drawing in paragraph.drawings:
                target = self.drawing_target(drawing)
                if target is None:
                    self.error(
                        f"Paragraph {paragraph.index + 1} contains an unresolved drawing."
                    )
                    continue
                try:
                    payload = package.read(target)
                except KeyError:
                    self.error(
                        f"Paragraph {paragraph.index + 1} drawing target is missing: {target}."
                    )
                    continue
                if paragraph.index in prompt_indices:
                    continue
                digest = sha256_bytes(payload)
                if digest not in allowed:
                    self.error(
                        f"Paragraph {paragraph.index + 1} contains unapproved answer "
                        f"media {target} with SHA-256 {digest}."
                    )

    def check_header(self, package: zipfile.ZipFile) -> None:
        header_names = sorted(
            name
            for name in package.namelist()
            if re.fullmatch(r"word/header\d+\.xml", name)
        )
        if not header_names:
            self.error("No Word header part was found.")
            return
        combined_text: list[str] = []
        tab_stops: set[tuple[str, int]] = set()
        for name in header_names:
            root = ET.fromstring(package.read(name))
            combined_text.append(
                clean_text("".join(node.text or "" for node in root.findall(".//w:t", NS)))
            )
            for tab in root.findall(".//w:tabs/w:tab", NS):
                value = wattr(tab, "val")
                position = wattr(tab, "pos")
                if value and position and position.lstrip("-").isdigit():
                    tab_stops.add((value, int(position)))
            for paragraph in root.findall(".//w:p", NS):
                ppr = paragraph.find("w:pPr", NS)
                style_id = wattr(
                    ppr.find("w:pStyle", NS) if ppr is not None else None, "val"
                ) or "Normal"
                _, style_rpr = self.effective_style(style_id)
                for run in paragraph.findall("w:r", NS):
                    run_text = "".join(
                        node.text or "" for node in run.findall(".//w:t", NS)
                    )
                    if not run_text.strip():
                        continue
                    effective_rpr = nested_merge(
                        style_rpr, rpr_values(run.find("w:rPr", NS))
                    )
                    fonts = set(effective_rpr.get("fonts", {}).values())
                    required_font = self.contract["header"]["font"]
                    if required_font not in fonts:
                        self.error(
                            f"Header run {run_text!r} does not resolve to "
                            f"{required_font!r}; fonts={sorted(fonts)!r}."
                        )
                    actual_size = effective_rpr.get("size_half_points")
                    required_size = self.contract["header"]["size_half_points"]
                    if actual_size != required_size:
                        self.error(
                            f"Header run {run_text!r} is {actual_size} half-points; "
                            f"required {required_size}."
                        )
        text = " ".join(combined_text)
        for fragment in self.contract["header"]["required_text_fragments"]:
            if fragment not in text:
                self.error(f"Header is missing required text fragment {fragment!r}.")
        for tab in self.contract["header"]["required_tab_stops"]:
            required = (tab["value"], tab["position_twips"])
            if required not in tab_stops:
                self.error(
                    f"Header is missing required tab stop {required[0]!r} at "
                    f"{required[1]} twips."
                )

    def check_table_of_contents(self, document_root: ET.Element) -> None:
        if not self.contract.get("table_of_contents", {}).get("required"):
            return
        field_codes = " ".join(
            node.text or "" for node in document_root.findall(".//w:instrText", NS)
        )
        if not re.search(r"\bTOC\b", field_codes, flags=re.IGNORECASE):
            self.error("The document has no Word table-of-contents field.")

    def run(self) -> int:
        if not self.docx_path.is_file():
            self.error(f"DOCX does not exist: {self.docx_path}")
        if not self.contract_path.is_file():
            self.error(f"Contract does not exist: {self.contract_path}")
        if self.errors:
            return self.report()

        try:
            with zipfile.ZipFile(self.docx_path) as package:
                required_parts = {
                    "word/document.xml",
                    "word/styles.xml",
                    "word/_rels/document.xml.rels",
                }
                missing = sorted(required_parts - set(package.namelist()))
                if missing:
                    self.error(f"DOCX is missing required package parts: {missing!r}.")
                    return self.report()
                self.load_styles(package)
                self.load_relationships(package)
                document_root = self.load_document(package)
                self.check_geometry(document_root)
                self.check_styles()
                self.check_table_of_contents(document_root)
                self.check_forbidden_styles()
                prompt_indices = self.check_semantic_structure()
                self.check_hebrew_paragraphs()
                self.check_equations()
                self.check_media(package, prompt_indices)
                self.check_header(package)
        except (zipfile.BadZipFile, ET.ParseError, KeyError, ValueError) as exc:
            self.error(f"Could not audit DOCX package: {exc}")

        return self.report()

    def report(self) -> int:
        digest = sha256_file(self.docx_path) if self.docx_path.is_file() else None
        baseline = self.contract.get("accepted_baseline", {})
        rejected = self.contract.get("known_rejected_checkpoint", {})
        classification = "unclassified"
        if digest == baseline.get("sha256"):
            classification = "accepted-baseline"
        elif digest == rejected.get("sha256"):
            classification = "known-rejected-checkpoint"

        status = "PASS" if not self.errors else "FAIL"
        print(f"{status}: {self.docx_path}")
        if digest:
            print(f"SHA-256: {digest}")
        print(f"Classification: {classification}")
        if self.errors:
            print(f"Violations: {len(self.errors)}")
            for number, message in enumerate(self.errors, start=1):
                print(f"  {number}. {message}")
            print(
                "This candidate must not be promoted. Fix the violations and "
                "then complete the separate Microsoft Word full-page visual gate."
            )
            return 1
        print(
            "Mechanical OOXML contract passed. Microsoft Word full-page visual "
            "review and explicit user approval are still required."
        )
        return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit HW4 DOCX structure, styles, RTL, prompts, equations, and media."
    )
    parser.add_argument("docx", type=Path)
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path("style/homework_style_contract.json"),
    )
    return parser.parse_args()


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args()
    return Auditor(args.docx.resolve(), args.contract.resolve()).run()


if __name__ == "__main__":
    sys.exit(main())
