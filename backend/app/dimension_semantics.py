from __future__ import annotations

import math
import re
from dataclasses import dataclass

import networkx as nx

from .cad_layers import DIMENSION, TEXT, canonical_layer_name
from .models import (
    DimensionBinding,
    Entity,
    LineEntity,
    OcrRegion,
    ParsedDimensionValue,
    ProjectState,
    RectangleEntity,
    TextEntity,
)

SEMANTIC_SOURCE = "dimension_semantics_v0"
MIN_DIMENSION_LINE_LENGTH_MM = 4.0
MIN_ARROW_LENGTH_MM = 0.45
MAX_ARROW_LENGTH_MM = 8.5
MAX_TAGGED_ARROW_LENGTH_MM = 12.0
ARROW_ENDPOINT_TOLERANCE_MM = 4.5
ARROW_MIN_ANGLE_RAD = math.radians(14)
ARROW_MAX_ANGLE_RAD = math.radians(130)
TEXT_BIND_DISTANCE_MM = 38.0
OCR_TEXT_BIND_DISTANCE_MM = 86.0
TEXT_ARROW_DISTANCE_MM = 55.0
OCR_TEXT_ARROW_DISTANCE_MM = 110.0
NUMBER_RE = r"\d+(?:\.\d+)?"


@dataclass(frozen=True)
class DimensionSemantics:
    bindings: list[DimensionBinding]
    warnings: list[str]


@dataclass(frozen=True)
class LineRef:
    entity: LineEntity
    x1: float
    y1: float
    x2: float
    y2: float
    length: float
    angle: float

    @property
    def midpoint(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) * 0.5, (self.y1 + self.y2) * 0.5)

    @property
    def start(self) -> tuple[float, float]:
        return (self.x1, self.y1)

    @property
    def end(self) -> tuple[float, float]:
        return (self.x2, self.y2)


@dataclass(frozen=True)
class ArrowMatch:
    line: LineRef
    endpoint: str
    distance: float
    angle_delta: float


@dataclass(frozen=True)
class TextCandidate:
    id: str
    text: str
    x: float
    y: float
    parsed: ParsedDimensionValue
    confidence: float
    source: str


@dataclass(frozen=True)
class GraphBindingCandidate:
    line: LineRef
    text: TextCandidate | None
    arrow_matches: list[ArrowMatch]
    text_distance: float | None
    score: float
    path: list[str]


@dataclass(frozen=True)
class Box:
    x: float
    y: float
    width: float
    height: float

    @property
    def center(self) -> tuple[float, float]:
        return (self.x + self.width * 0.5, self.y + self.height * 0.5)


class DimensionGraphBuilder:
    """Build a graph for text -> arrowhead -> dimension-line binding."""

    def __init__(
        self,
        dimension_lines: list[LineRef],
        arrow_lines: list[LineRef],
        text_candidates: list[TextCandidate],
    ) -> None:
        self.dimension_lines = dimension_lines
        self.arrow_lines = arrow_lines
        self.text_candidates = text_candidates
        self.graph = nx.Graph()
        self.line_nodes: dict[str, LineRef] = {}
        self.arrow_nodes: dict[str, LineRef] = {}
        self.text_nodes: dict[str, TextCandidate] = {}

    def build(self) -> nx.Graph:
        for line in self.dimension_lines:
            node_id = self._line_node(line)
            self.line_nodes[node_id] = line
            self.graph.add_node(node_id, kind="dimension_line", ref=line)
        for arrow in self.arrow_lines:
            node_id = self._arrow_node(arrow)
            self.arrow_nodes[node_id] = arrow
            self.graph.add_node(node_id, kind="arrow", ref=arrow)
        for text in self.text_candidates:
            node_id = self._text_node(text)
            self.text_nodes[node_id] = text
            self.graph.add_node(node_id, kind="text", ref=text)

        self._add_arrow_line_edges()
        self._add_text_arrow_edges()
        return self.graph

    def binding_candidates(self) -> list[GraphBindingCandidate]:
        if not self.graph.nodes:
            self.build()

        candidates: list[GraphBindingCandidate] = []
        for text_node, text in self.text_nodes.items():
            for line_node, line in self.line_nodes.items():
                try:
                    path = nx.shortest_path(self.graph, text_node, line_node, weight="weight")
                    score = float(nx.path_weight(self.graph, path, weight="weight"))
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    continue
                arrows = [
                    self.arrow_nodes[node]
                    for node in path
                    if node in self.arrow_nodes
                ]
                if not arrows:
                    continue
                arrow_matches = self._line_arrow_matches(line_node)
                if not arrow_matches:
                    continue
                text_distance = self._path_text_arrow_distance(path)
                if text.id.startswith("ocr_"):
                    score += 4.0
                if text.parsed.kind == "unknown":
                    score += 3.0
                candidates.append(
                    GraphBindingCandidate(
                        line=line,
                        text=text,
                        arrow_matches=arrow_matches,
                        text_distance=text_distance,
                        score=score,
                        path=path,
                    )
                )

        bound_line_ids = {candidate.line.entity.id for candidate in candidates}
        for line_node, line in self.line_nodes.items():
            if line.entity.id in bound_line_ids:
                continue
            arrow_matches = self._line_arrow_matches(line_node)
            if not arrow_matches:
                continue
            candidates.append(
                GraphBindingCandidate(
                    line=line,
                    text=None,
                    arrow_matches=arrow_matches,
                    text_distance=None,
                    score=100.0 + min(match.distance for match in arrow_matches),
                    path=[line_node, *[self._arrow_node(match.line) for match in arrow_matches]],
                )
            )
        return sorted(candidates, key=lambda item: item.score)

    def _add_arrow_line_edges(self) -> None:
        for line_node, line in self.line_nodes.items():
            for arrow_node, arrow in self.arrow_nodes.items():
                match = _arrow_match(line, arrow)
                if match is None:
                    continue
                weight = 1.0 + match.distance / max(ARROW_ENDPOINT_TOLERANCE_MM, 1e-6)
                self.graph.add_edge(
                    arrow_node,
                    line_node,
                    kind="arrow_to_dimension_line",
                    weight=weight,
                    endpoint=match.endpoint,
                    distance=match.distance,
                    angle_delta=match.angle_delta,
                )

    def _add_text_arrow_edges(self) -> None:
        for text_node, text in self.text_nodes.items():
            limit = OCR_TEXT_ARROW_DISTANCE_MM if text.id.startswith("ocr_") else TEXT_ARROW_DISTANCE_MM
            for arrow_node, arrow in self.arrow_nodes.items():
                distance = _text_arrow_distance(text, arrow)
                if distance > limit:
                    continue
                weight = 1.0 + distance / max(limit, 1e-6)
                self.graph.add_edge(
                    text_node,
                    arrow_node,
                    kind="text_to_arrow",
                    weight=weight,
                    distance=distance,
                )

    def _line_arrow_matches(self, line_node: str) -> list[ArrowMatch]:
        matches: list[ArrowMatch] = []
        for neighbor in self.graph.neighbors(line_node):
            if neighbor not in self.arrow_nodes:
                continue
            edge = self.graph.edges[line_node, neighbor]
            arrow = self.arrow_nodes[neighbor]
            matches.append(
                ArrowMatch(
                    line=arrow,
                    endpoint=str(edge.get("endpoint") or "unknown"),
                    distance=float(edge.get("distance") or 0.0),
                    angle_delta=float(edge.get("angle_delta") or 0.0),
                )
            )
        return sorted(matches, key=lambda item: (item.distance, item.line.entity.id))[:4]

    def _path_text_arrow_distance(self, path: list[str]) -> float | None:
        distances = [
            float(self.graph.edges[first, second].get("distance") or 0.0)
            for first, second in zip(path, path[1:])
            if self.graph.edges[first, second].get("kind") == "text_to_arrow"
        ]
        return min(distances) if distances else None

    @staticmethod
    def _line_node(line: LineRef) -> str:
        return f"line:{line.entity.id}"

    @staticmethod
    def _arrow_node(arrow: LineRef) -> str:
        return f"arrow:{arrow.entity.id}"

    @staticmethod
    def _text_node(text: TextCandidate) -> str:
        return f"text:{text.id}"


def detect_dimension_bindings(project: ProjectState) -> DimensionSemantics:
    """Bind dimension lines, arrow strokes, and nearby dimension text.

    The baseline is intentionally conservative and explainable:
    1. collect dimension-line candidates from promoted/dimension semantic layers,
    2. pair short angled arrow strokes to candidate line endpoints,
    3. bind the nearest parsed dimension text around the line.
    """
    dimension_lines, arrow_lines = _collect_line_candidates(project.ir.entities)
    text_candidates = _collect_text_candidates(project)
    warnings: list[str] = []
    if not dimension_lines:
        warnings.append("No dimension-line candidates found. Run vector extraction or scan promotion first.")
    if not arrow_lines:
        warnings.append("No arrowhead candidates found near promoted/dimension geometry.")
    if not text_candidates:
        warnings.append("No dimension text candidates found. Run OCR or use a vector PDF with extractable text.")

    graph_builder = DimensionGraphBuilder(dimension_lines, arrow_lines, text_candidates)
    graph_builder.build()
    graph_candidates = graph_builder.binding_candidates()

    bindings: list[DimensionBinding] = []
    used_line_ids: set[str] = set()
    used_text_ids: set[str] = set()
    unbound_arrowed_lines = 0
    for candidate in graph_candidates:
        line = candidate.line
        if line.entity.id in used_line_ids:
            continue
        text_candidate = candidate.text
        if text_candidate is not None and text_candidate.id in used_text_ids:
            continue
        arrow_matches = candidate.arrow_matches
        if text_candidate is None:
            unbound_arrowed_lines += 1
            parsed = ParsedDimensionValue(raw_text="")
        else:
            parsed = text_candidate.parsed
            used_text_ids.add(text_candidate.id)
        used_line_ids.add(line.entity.id)

        confidence = _binding_confidence(arrow_matches, text_candidate, candidate.text_distance)
        bindings.append(
            DimensionBinding(
                id=f"dim_binding_{len(bindings):05d}",
                dimension_line_id=line.entity.id,
                arrow_ids=[match.line.entity.id for match in arrow_matches],
                text_id=text_candidate.id if text_candidate else None,
                text=text_candidate.text if text_candidate else "",
                parsed=parsed,
                confidence=confidence,
                kind=parsed.kind,
                line_x1=line.x1,
                line_y1=line.y1,
                line_x2=line.x2,
                line_y2=line.y2,
                text_x=text_candidate.x if text_candidate else None,
                text_y=text_candidate.y if text_candidate else None,
                binding_method="graph_text_arrow_line" if text_candidate else "graph_arrow_line",
                graph_path=candidate.path,
                graph_score=round(candidate.score, 3),
                source=SEMANTIC_SOURCE,
            )
        )

    bindings.sort(key=lambda item: item.confidence, reverse=True)
    for index, binding in enumerate(bindings):
        binding.id = f"dim_binding_{index:05d}"
    if unbound_arrowed_lines:
        warnings.append(f"{unbound_arrowed_lines} arrowed dimension lines had no nearby dimension text.")
    if arrow_lines and not bindings:
        warnings.append("Arrowheads were found, but no line-arrow-text bindings passed the distance thresholds.")
    return DimensionSemantics(bindings=bindings, warnings=warnings)


def parse_dimension_text(text: str) -> ParsedDimensionValue:
    raw = text.strip()
    normalized = _normalize_dimension_text(raw)
    if not normalized:
        return ParsedDimensionValue(raw_text=raw)

    roughness = re.search(rf"\bRa\s*({NUMBER_RE})", normalized, flags=re.IGNORECASE)
    if roughness:
        return ParsedDimensionValue(
            kind="roughness",
            raw_text=raw,
            nominal=float(roughness.group(1)),
            unit="um",
        )

    diameter = re.search(rf"[φΦØ⌀]\s*({NUMBER_RE})", normalized)
    if diameter:
        upper, lower = _parse_tolerances(normalized, diameter.end())
        return ParsedDimensionValue(
            kind="diameter",
            raw_text=raw,
            nominal=float(diameter.group(1)),
            upper_tol=upper,
            lower_tol=lower,
        )

    radius = re.search(rf"(?<![A-Za-z])R\s*({NUMBER_RE})", normalized, flags=re.IGNORECASE)
    if radius:
        upper, lower = _parse_tolerances(normalized, radius.end())
        return ParsedDimensionValue(
            kind="radius",
            raw_text=raw,
            nominal=float(radius.group(1)),
            upper_tol=upper,
            lower_tol=lower,
        )

    first_number = re.search(NUMBER_RE, normalized)
    if first_number:
        upper, lower = _parse_tolerances(normalized, first_number.end())
        kind = "tolerance" if _looks_like_geometric_tolerance(normalized, first_number) else "linear"
        return ParsedDimensionValue(
            kind=kind,
            raw_text=raw,
            nominal=float(first_number.group(0)),
            upper_tol=upper,
            lower_tol=lower,
        )

    return ParsedDimensionValue(raw_text=raw)


def _collect_line_candidates(entities: list[Entity]) -> tuple[list[LineRef], list[LineRef]]:
    dimension_lines: list[LineRef] = []
    arrow_lines: list[LineRef] = []
    for entity in entities:
        if not isinstance(entity, LineEntity):
            continue
        ref = _line_ref(entity)
        if ref is None:
            continue
        if _is_arrow_candidate(entity, ref):
            arrow_lines.append(ref)
            continue
        if _is_dimension_line_candidate(entity, ref):
            dimension_lines.append(ref)
    return dimension_lines, arrow_lines


def _line_ref(entity: LineEntity) -> LineRef | None:
    length = _distance((entity.x1, entity.y1), (entity.x2, entity.y2))
    if length <= 1e-9:
        return None
    angle = math.atan2(entity.y2 - entity.y1, entity.x2 - entity.x1) % math.pi
    return LineRef(
        entity=entity,
        x1=float(entity.x1),
        y1=float(entity.y1),
        x2=float(entity.x2),
        y2=float(entity.y2),
        length=length,
        angle=angle,
    )


def _is_arrow_candidate(entity: LineEntity, line: LineRef) -> bool:
    tags = set(entity.tags)
    if "dimension_arrow_render" in tags:
        return False
    tagged_arrow = bool(tags.intersection({"dimension_arrow", "arrowhead"}))
    if tagged_arrow:
        return MIN_ARROW_LENGTH_MM <= line.length <= MAX_TAGGED_ARROW_LENGTH_MM
    if not _is_dimension_semantic_entity(entity):
        return False
    if tags.intersection({"hatch", "cut_hatch", "centerline"}):
        return False
    return MIN_ARROW_LENGTH_MM <= line.length <= MAX_ARROW_LENGTH_MM and _is_diagonal(line)


def _is_dimension_line_candidate(entity: LineEntity, line: LineRef) -> bool:
    if line.length < MIN_DIMENSION_LINE_LENGTH_MM:
        return False
    if set(entity.tags).intersection({"dimension_arrow", "arrowhead", "hatch", "cut_hatch"}):
        return False
    if (
        entity.group == "promoted_geometry"
        or entity.layer == "promoted_geometry"
        or entity.metadata.get("legacy_layer") == "promoted_geometry"
    ):
        return True
    return _is_dimension_semantic_entity(entity)


def _is_dimension_semantic_entity(entity: LineEntity) -> bool:
    return canonical_layer_name(entity.layer) == DIMENSION or entity.group == "dimensions" or "dimensions" in entity.tags


def _is_diagonal(line: LineRef) -> bool:
    dx = abs(line.x2 - line.x1)
    dy = abs(line.y2 - line.y1)
    if dx <= 1e-9 or dy <= 1e-9:
        return False
    slope = dy / dx
    return 0.18 <= slope <= 5.6


def _match_arrows_to_line(line: LineRef, arrows: list[LineRef]) -> list[ArrowMatch]:
    matches: list[ArrowMatch] = []
    for arrow in arrows:
        match = _arrow_match(line, arrow)
        if match is not None:
            matches.append(match)
    matches.sort(key=lambda item: (item.distance, item.line.entity.id))

    selected: list[ArrowMatch] = []
    seen: set[str] = set()
    for match in matches:
        if match.line.entity.id in seen:
            continue
        selected.append(match)
        seen.add(match.line.entity.id)
        if len(selected) >= 4:
            break
    return selected


def _arrow_match(line: LineRef, arrow: LineRef) -> ArrowMatch | None:
    endpoint, distance = _nearest_line_arrow_endpoint(line, arrow)
    if distance > ARROW_ENDPOINT_TOLERANCE_MM:
        return None
    angle_delta = _angle_delta(line.angle, arrow.angle)
    if not (ARROW_MIN_ANGLE_RAD <= angle_delta <= ARROW_MAX_ANGLE_RAD):
        return None
    return ArrowMatch(line=arrow, endpoint=endpoint, distance=distance, angle_delta=angle_delta)


def _nearest_line_arrow_endpoint(line: LineRef, arrow: LineRef) -> tuple[str, float]:
    candidates = [
        ("start", _distance(line.start, arrow.start)),
        ("start", _distance(line.start, arrow.end)),
        ("end", _distance(line.end, arrow.start)),
        ("end", _distance(line.end, arrow.end)),
    ]
    return min(candidates, key=lambda item: item[1])


def _collect_text_candidates(project: ProjectState) -> list[TextCandidate]:
    candidates: list[TextCandidate] = []
    for entity in project.ir.entities:
        if not isinstance(entity, TextEntity):
            continue
        text = entity.text.strip()
        if not text:
            continue
        parsed = parse_dimension_text(text)
        if parsed.kind == "unknown" and not _entity_text_lives_in_dimension_area(entity):
            continue
        if parsed.kind == "unknown" and not re.search(NUMBER_RE, _normalize_dimension_text(text)):
            continue
        candidates.append(
            TextCandidate(
                id=entity.id,
                text=text,
                x=float(entity.x),
                y=float(entity.y),
                parsed=parsed,
                confidence=0.9,
                source="text_entity",
            )
        )

    sheet = _sheet_box(project.ir.entities)
    for region in project.ocr_regions:
        if region.target != "dimensions" or not region.text.strip():
            continue
        candidates.extend(_ocr_region_text_candidates(region, sheet, len(candidates)))
    return candidates


def _entity_text_lives_in_dimension_area(entity: TextEntity) -> bool:
    return canonical_layer_name(entity.layer) in {DIMENSION, TEXT} or entity.group == "dimensions" or "dimensions" in entity.tags


def _ocr_region_text_candidates(region: OcrRegion, sheet: Box, offset: int) -> list[TextCandidate]:
    center_x = sheet.x + (region.x + region.width * 0.5) * sheet.width
    center_y = sheet.y + (1.0 - region.y - region.height * 0.5) * sheet.height
    fragments = _dimension_fragments(region.text)
    if not fragments and re.search(NUMBER_RE, region.text):
        fragments = [region.text.strip()]

    candidates: list[TextCandidate] = []
    for index, fragment in enumerate(fragments):
        parsed = parse_dimension_text(fragment)
        candidates.append(
            TextCandidate(
                id=f"ocr_dimensions_{offset + index:05d}",
                text=fragment,
                x=center_x,
                y=center_y,
                parsed=parsed,
                confidence=region.confidence,
                source=region.source or "ocr_region",
            )
        )
    return candidates


def _dimension_fragments(text: str) -> list[str]:
    normalized = _normalize_dimension_text(text)
    pattern = re.compile(
        rf"(?:Ra\s*{NUMBER_RE}|[φΦØ⌀]\s*{NUMBER_RE}(?:\s*(?:±|[+]|[-])\s*{NUMBER_RE})?"
        rf"(?:\s*[-]\s*{NUMBER_RE})?|R\s*{NUMBER_RE}|{NUMBER_RE}\s*(?:±\s*{NUMBER_RE}|[+]\s*{NUMBER_RE}"
        rf"(?:\s*[-]\s*{NUMBER_RE})?|[-]\s*{NUMBER_RE})?(?:\s*[A-Z])?)",
        flags=re.IGNORECASE,
    )
    fragments: list[str] = []
    seen: set[str] = set()
    for match in pattern.finditer(normalized):
        fragment = match.group(0).strip()
        if not fragment or fragment in seen:
            continue
        seen.add(fragment)
        fragments.append(fragment)
    return fragments[:60]


def _nearest_text_candidate(
    line: LineRef,
    candidates: list[TextCandidate],
    used_text_ids: set[str],
) -> tuple[TextCandidate | None, float | None]:
    best: TextCandidate | None = None
    best_distance = math.inf
    for candidate in candidates:
        score = _text_line_distance(line, candidate)
        limit = OCR_TEXT_BIND_DISTANCE_MM if candidate.id.startswith("ocr_") else TEXT_BIND_DISTANCE_MM
        if candidate.parsed.kind == "unknown":
            score += 12.0
        if candidate.id in used_text_ids:
            score += 18.0
        if score < best_distance and score <= limit:
            best = candidate
            best_distance = score
    return best, best_distance if best is not None else None


def _text_line_distance(line: LineRef, text: TextCandidate) -> float:
    midpoint_distance = _distance(line.midpoint, (text.x, text.y))
    segment_distance = _point_to_segment_distance((text.x, text.y), line.start, line.end)
    return midpoint_distance * 0.45 + segment_distance * 0.55


def _text_arrow_distance(text: TextCandidate, arrow: LineRef) -> float:
    point = (text.x, text.y)
    midpoint = arrow.midpoint
    return min(
        _distance(point, arrow.start),
        _distance(point, arrow.end),
        _distance(point, midpoint),
    )


def _binding_confidence(
    arrow_matches: list[ArrowMatch],
    text_candidate: TextCandidate | None,
    text_distance: float | None,
) -> float:
    endpoint_count = len({match.endpoint for match in arrow_matches})
    confidence = 0.28 + min(len(arrow_matches), 4) * 0.07 + endpoint_count * 0.10
    if text_candidate is not None:
        limit = OCR_TEXT_BIND_DISTANCE_MM if text_candidate.id.startswith("ocr_") else TEXT_BIND_DISTANCE_MM
        distance_score = 1.0 - min(text_distance or limit, limit) / max(limit, 1e-6)
        confidence += 0.18 + distance_score * 0.18
        if text_candidate.parsed.kind != "unknown":
            confidence += 0.17
        if text_candidate.id.startswith("ocr_"):
            confidence -= 0.08
    return round(max(0.0, min(confidence, 0.99)), 3)


def _sheet_box(entities: list[Entity]) -> Box:
    rectangles = [entity for entity in entities if isinstance(entity, RectangleEntity)]
    preferred = [
        entity
        for entity in rectangles
        if entity.id in {"scan_sheet_border", "sheet_border"} or entity.group == "sheet" or "sheet" in entity.tags
    ]
    if preferred:
        rect = max(preferred, key=lambda item: item.width * item.height)
        return Box(float(rect.x), float(rect.y), float(rect.width), float(rect.height))
    if rectangles:
        rect = max(rectangles, key=lambda item: item.width * item.height)
        return Box(float(rect.x), float(rect.y), float(rect.width), float(rect.height))

    points: list[tuple[float, float]] = []
    for entity in entities:
        if isinstance(entity, LineEntity):
            points.extend([(entity.x1, entity.y1), (entity.x2, entity.y2)])
        elif isinstance(entity, TextEntity):
            points.append((entity.x, entity.y))
    if not points:
        return Box(0.0, 0.0, 420.0, 297.0)
    min_x = min(point[0] for point in points)
    min_y = min(point[1] for point in points)
    max_x = max(point[0] for point in points)
    max_y = max(point[1] for point in points)
    return Box(float(min_x), float(min_y), float(max(max_x - min_x, 1.0)), float(max(max_y - min_y, 1.0)))


def _parse_tolerances(text: str, start: int) -> tuple[float | None, float | None]:
    suffix = text[start : start + 42]
    plus_minus = re.search(rf"±\s*({NUMBER_RE})", suffix)
    if plus_minus:
        value = float(plus_minus.group(1))
        return value, -value

    signed_values = [
        _signed_number_value(match.group(0))
        for match in re.finditer(rf"[+\-]\s*{NUMBER_RE}", suffix)
    ]
    if signed_values:
        upper = next((value for value in signed_values if value >= 0), None)
        lower = next((value for value in signed_values if value < 0), None)
        if upper is None and re.match(rf"\s*0(?:\.0+)?\s*-\s*{NUMBER_RE}", suffix):
            upper = 0.0
        return upper, lower
    return None, None


def _signed_number_value(text: str) -> float:
    compact = text.replace(" ", "")
    return float(compact)


def _looks_like_geometric_tolerance(text: str, first_number: re.Match[str]) -> bool:
    nominal = float(first_number.group(0))
    suffix = text[first_number.end() : first_number.end() + 12].strip()
    return nominal < 1.0 and bool(re.match(r"^[A-Z]?$", suffix))


def _normalize_dimension_text(text: str) -> str:
    return (
        text.replace("＋", "+")
        .replace("－", "-")
        .replace("−", "-")
        .replace("—", "-")
        .replace("Φ", "φ")
        .replace("Ø", "φ")
        .replace("⌀", "φ")
        .replace("：", ":")
        .strip()
    )


def _distance(first: tuple[float, float], second: tuple[float, float]) -> float:
    return math.hypot(first[0] - second[0], first[1] - second[1])


def _point_to_segment_distance(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> float:
    vx = end[0] - start[0]
    vy = end[1] - start[1]
    length_sq = vx * vx + vy * vy
    if length_sq <= 1e-12:
        return _distance(point, start)
    t = ((point[0] - start[0]) * vx + (point[1] - start[1]) * vy) / length_sq
    t = max(0.0, min(1.0, t))
    projection = (start[0] + t * vx, start[1] + t * vy)
    return _distance(point, projection)


def _angle_delta(first: float, second: float) -> float:
    delta = abs(first - second) % math.pi
    return min(delta, math.pi - delta)
