"""
multimodal.py
=============

Perception layer — converts non-text inputs into structured
representations the reasoning system can work with.

The architecture keeps perception separate from reasoning.
The perception layer converts images, audio, and documents into
rich structured text. The reasoning system then works with that
text exactly as it works with any other input. The small model
doesn't need vision — the perception layer provides vision.

Three perceivers:

  ImagePerceiver     — understands images using the Anthropic Vision
                       API. Extracts: semantic description, scientific
                       diagram structure, visible text (OCR), numerical
                       values from plots, spatial relationships.
                       Falls back to structural analysis (EXIF, format)
                       if no vision API is available.

  AudioPerceiver     — transcribes audio using OpenAI Whisper
                       (runs locally, no API key needed, ~1.5GB).
                       Identifies language, extracts key segments,
                       notes tone and context.

  DocumentPerceiver  — extracts text and structure from PDFs and
                       documents. Identifies: sections, figures,
                       tables, mathematical notation, citations.

  PerceptionHub      — coordinates all perceivers, auto-detects
                       input type, and formats the result as context
                       for injection into the reasoning chain.

Perception results format as:
  [PERCEPTION: filename]
    Type: scientific diagram | photograph | audio | document
    Content: ...structured description...
    Key values: ...extracted numerical or textual data...
    Relevance to question: ...
  [/PERCEPTION]
"""

import os
import re
import json
import base64
import time
import mimetypes
from dataclasses import dataclass, field
from typing import Optional, Callable
from pathlib import Path


# ============================================================
# PerceptionResult
# ============================================================

@dataclass
class PerceptionResult:
    input_type:   str       # "image" | "audio" | "document" | "unknown"
    source:       str       # filename or URL
    description:  str       # semantic description
    extracted:    dict      # structured extracted data
    raw_text:     str       # any text extracted verbatim
    confidence:   float     # 0-1 confidence in perception quality
    elapsed:      float     = 0.0
    error:        str       = ""

    def format_for_reasoning(self, question: str = "") -> str:
        """Format as context for injection into the Reasoner."""
        if self.error:
            return f"[PERCEPTION: {self.source}]\n  Error: {self.error}\n[/PERCEPTION]"

        lines = [f"[PERCEPTION: {Path(self.source).name}]"]
        lines.append(f"  Type: {self.input_type}")

        if self.description:
            lines.append(f"  Description: {self.description[:300]}")

        for key, val in self.extracted.items():
            if val:
                val_str = str(val)[:150]
                lines.append(f"  {key.replace('_', ' ').title()}: {val_str}")

        if self.raw_text:
            excerpt = self.raw_text[:200]
            lines.append(f"  Extracted text: {excerpt}")

        if question:
            lines.append(f"  (Relevant to: {question[:80]})")

        lines.append(f"  [confidence: {self.confidence:.2f}]")
        lines.append("[/PERCEPTION]")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "input_type":  self.input_type,
            "source":      self.source,
            "description": self.description[:300],
            "extracted":   self.extracted,
            "confidence":  self.confidence,
        }


# ============================================================
# ImagePerceiver
# ============================================================

IMAGE_PERCEPTION_SYSTEM = """You analyse images and describe them with scientific precision.

For scientific diagrams (plots, graphs, charts):
  - Identify axis labels and units
  - Extract key numerical values
  - Describe the shape and trend of any curves
  - Note any critical points, thresholds, or anomalies

For photographs:
  - Describe what is shown
  - Note spatial relationships
  - Extract any visible text or numbers

For mathematical or symbolic content:
  - Transcribe equations and symbols precisely
  - Describe the mathematical structure

Output as structured text. Be precise, not verbose."""

IMAGE_PERCEPTION_USER = """Describe this image in structured detail.
{question_context}
Focus on information that would be useful for reasoning about the question."""


class ImagePerceiver:
    """
    Understands images using the Anthropic Vision API.

    Falls back to structural/metadata analysis if the API is unavailable.
    """

    def __init__(
        self,
        api_key:    str  = "",
        verbose:    bool = True,
    ):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.verbose = verbose

    def perceive(
        self,
        image_source: str,    # path or URL
        question:     str = "",
    ) -> PerceptionResult:
        """
        Perceive an image and return a structured description.
        """
        t0 = time.time()
        source = image_source

        # Load image
        image_data, media_type = self._load_image(image_source)
        if image_data is None:
            return PerceptionResult(
                input_type="image", source=source,
                description="", extracted={}, raw_text="",
                confidence=0.0, elapsed=time.time()-t0,
                error="Could not load image",
            )

        # Try Vision API
        if self.api_key:
            return self._perceive_with_api(
                image_data, media_type, source, question, t0
            )

        # Fallback: structural analysis
        return self._structural_analysis(image_source, source, t0)

    def _load_image(self, source: str):
        """Load image as base64. Returns (base64_data, media_type) or (None, None)."""
        try:
            if source.startswith(("http://", "https://")):
                import urllib.request
                with urllib.request.urlopen(source, timeout=10) as r:
                    data = r.read()
                    mt   = r.headers.get("content-type", "image/jpeg").split(";")[0]
                    return base64.standard_b64encode(data).decode(), mt
            else:
                if not os.path.exists(source):
                    return None, None
                mt = mimetypes.guess_type(source)[0] or "image/jpeg"
                with open(source, "rb") as f:
                    return base64.standard_b64encode(f.read()).decode(), mt
        except Exception:
            return None, None

    def _perceive_with_api(
        self,
        image_data: str,
        media_type: str,
        source:     str,
        question:   str,
        t0:         float,
    ) -> PerceptionResult:
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=self.api_key)

            q_ctx = f"\nQuestion being asked: {question}" if question else ""
            user  = IMAGE_PERCEPTION_USER.format(question_context=q_ctx)

            response = client.messages.create(
                model="claude-opus-4-5",
                max_tokens=800,
                system=IMAGE_PERCEPTION_SYSTEM,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type":       "base64",
                                "media_type": media_type,
                                "data":       image_data,
                            },
                        },
                        {"type": "text", "text": user},
                    ],
                }],
            )

            description = response.content[0].text

            # Extract structured data from description
            extracted = self._extract_structured(description)

            if self.verbose:
                print(f"  [image] Perceived {Path(source).name}: {description[:60]}...")

            return PerceptionResult(
                input_type="image",
                source=source,
                description=description,
                extracted=extracted,
                raw_text=self._extract_visible_text(description),
                confidence=0.90,
                elapsed=round(time.time()-t0, 2),
            )
        except Exception as e:
            return PerceptionResult(
                input_type="image", source=source,
                description="", extracted={}, raw_text="",
                confidence=0.0, elapsed=time.time()-t0,
                error=f"API error: {str(e)[:80]}",
            )

    def _structural_analysis(
        self, image_source: str, source: str, t0: float
    ) -> PerceptionResult:
        """Fallback: extract metadata without vision API."""
        extracted = {}
        try:
            from PIL import Image
            img = Image.open(image_source)
            extracted["dimensions"] = f"{img.width}×{img.height}px"
            extracted["mode"]       = img.mode
            extracted["format"]     = img.format or Path(image_source).suffix
        except ImportError:
            extracted["format"] = Path(image_source).suffix
            extracted["note"]   = "Install Pillow for image metadata"
        except Exception as e:
            extracted["error"] = str(e)[:60]

        return PerceptionResult(
            input_type="image",
            source=source,
            description=(
                "Image received. Set ANTHROPIC_API_KEY for full visual perception. "
                f"Format: {extracted.get('format', 'unknown')}, "
                f"dimensions: {extracted.get('dimensions', 'unknown')}"
            ),
            extracted=extracted,
            raw_text="",
            confidence=0.2,
            elapsed=round(time.time()-t0, 2),
        )

    @staticmethod
    def _extract_structured(text: str) -> dict:
        """Extract structured data from a vision description."""
        extracted = {}

        # Numerical values
        numbers = re.findall(r"(\w[\w\s]*?):\s*([\d.]+\s*(?:[a-zA-Z/²³°μ]+)?)", text)
        if numbers:
            extracted["numerical_values"] = {k.strip(): v.strip()
                                              for k, v in numbers[:5]}

        # Axis labels
        axes = re.findall(r"(?:x|horizontal|y|vertical)\s+axis[:\s]+([^\n.]+)", text, re.I)
        if axes:
            extracted["axes"] = axes[:2]

        # Trends
        trends = re.findall(r"(flat|rising|falling|decreasing|increasing|constant|linear|exponential|plateau)", text, re.I)
        if trends:
            extracted["trends"] = list(set(t.lower() for t in trends))

        # Scientific notation
        sci_vals = re.findall(r"[\d.]+\s*×\s*10[⁻⁰¹²³⁴⁵⁶⁷⁸⁹−-]+|\d+e[+-]?\d+", text)
        if sci_vals:
            extracted["scientific_values"] = sci_vals[:3]

        return extracted

    @staticmethod
    def _extract_visible_text(description: str) -> str:
        """Extract any text that was visibly present in the image."""
        m = re.search(
            r"(?:text|label|caption|inscription)[s]?[:\s]+([^\n]{10,})",
            description, re.I
        )
        return m.group(1).strip()[:200] if m else ""


# ============================================================
# AudioPerceiver
# ============================================================

class AudioPerceiver:
    """
    Transcribes audio using OpenAI Whisper (runs locally).

    Whisper is free, open-source, and runs on CPU.
    Install: pip install openai-whisper
    First run downloads the model (~150MB for 'base', ~1.5GB for 'large').
    """

    def __init__(
        self,
        model_size: str  = "base",   # tiny | base | small | medium | large
        verbose:    bool = True,
    ):
        self.model_size = model_size
        self.verbose    = verbose
        self._model     = None

    def _load_model(self):
        if self._model is None:
            try:
                import whisper
                self._model = whisper.load_model(self.model_size)
                if self.verbose:
                    print(f"  [audio] Whisper '{self.model_size}' loaded")
            except ImportError:
                raise ImportError(
                    "Install Whisper: pip install openai-whisper"
                )
        return self._model

    def perceive(
        self,
        audio_path: str,
        question:   str = "",
    ) -> PerceptionResult:
        """Transcribe audio and return structured result."""
        t0 = time.time()

        if not os.path.exists(audio_path):
            return PerceptionResult(
                input_type="audio", source=audio_path,
                description="", extracted={}, raw_text="",
                confidence=0.0, elapsed=time.time()-t0,
                error="File not found",
            )

        try:
            model  = self._load_model()
            result = model.transcribe(audio_path, fp16=False)

            transcript   = result.get("text", "").strip()
            language     = result.get("language", "unknown")
            segments     = result.get("segments", [])
            duration     = segments[-1]["end"] if segments else 0

            description = (
                f"{language.upper()} audio, {duration:.0f}s. "
                f"Transcript: {transcript[:200]}"
            )

            extracted = {
                "language":     language,
                "duration_sec": round(duration, 1),
                "n_segments":   len(segments),
                "word_count":   len(transcript.split()),
            }

            if self.verbose:
                print(f"  [audio] Transcribed {Path(audio_path).name}: "
                      f"{len(transcript)} chars, language={language}")

            return PerceptionResult(
                input_type="audio",
                source=audio_path,
                description=description,
                extracted=extracted,
                raw_text=transcript,
                confidence=0.90,
                elapsed=round(time.time()-t0, 2),
            )
        except Exception as e:
            return PerceptionResult(
                input_type="audio", source=audio_path,
                description="", extracted={}, raw_text="",
                confidence=0.0, elapsed=time.time()-t0,
                error=str(e)[:100],
            )


# ============================================================
# DocumentPerceiver
# ============================================================

class DocumentPerceiver:
    """
    Extracts text and structure from documents (PDF, DOCX, TXT).

    For PDFs: uses PyMuPDF (pip install pymupdf) or pdfminer.
    For DOCX: uses python-docx (pip install python-docx).
    For TXT:  reads directly.
    """

    def perceive(
        self,
        doc_path: str,
        question: str = "",
        max_chars: int = 3000,
    ) -> PerceptionResult:
        t0  = time.time()
        ext = Path(doc_path).suffix.lower()

        if ext == ".pdf":
            return self._perceive_pdf(doc_path, question, max_chars, t0)
        elif ext in (".docx", ".doc"):
            return self._perceive_docx(doc_path, question, max_chars, t0)
        elif ext in (".txt", ".md", ".rst"):
            return self._perceive_text(doc_path, question, max_chars, t0)
        else:
            return PerceptionResult(
                input_type="document", source=doc_path,
                description="", extracted={}, raw_text="",
                confidence=0.0, elapsed=time.time()-t0,
                error=f"Unsupported format: {ext}",
            )

    def _perceive_pdf(self, path, question, max_chars, t0) -> PerceptionResult:
        try:
            import fitz   # PyMuPDF
            doc   = fitz.open(path)
            text  = ""
            for page in doc:
                text += page.get_text()
                if len(text) > max_chars:
                    break
            doc.close()

            extracted = self._extract_document_structure(text)
            return PerceptionResult(
                input_type="document", source=path,
                description=f"PDF document, {len(text)} chars extracted",
                extracted=extracted,
                raw_text=text[:max_chars],
                confidence=0.85,
                elapsed=round(time.time()-t0, 2),
            )
        except ImportError:
            # Try pdfminer
            try:
                from pdfminer.high_level import extract_text
                text = extract_text(path)[:max_chars]
                return PerceptionResult(
                    input_type="document", source=path,
                    description=f"PDF document, {len(text)} chars",
                    extracted=self._extract_document_structure(text),
                    raw_text=text,
                    confidence=0.80,
                    elapsed=round(time.time()-t0, 2),
                )
            except ImportError:
                return PerceptionResult(
                    input_type="document", source=path,
                    description="PDF received but no PDF library available",
                    extracted={}, raw_text="",
                    confidence=0.0, elapsed=time.time()-t0,
                    error="Install: pip install pymupdf",
                )
        except Exception as e:
            return PerceptionResult(
                input_type="document", source=path,
                description="", extracted={}, raw_text="",
                confidence=0.0, elapsed=time.time()-t0,
                error=str(e)[:100],
            )

    def _perceive_docx(self, path, question, max_chars, t0) -> PerceptionResult:
        try:
            import docx
            doc   = docx.Document(path)
            text  = "\n".join(p.text for p in doc.paragraphs)[:max_chars]
            return PerceptionResult(
                input_type="document", source=path,
                description=f"Word document, {len(text)} chars",
                extracted=self._extract_document_structure(text),
                raw_text=text,
                confidence=0.85,
                elapsed=round(time.time()-t0, 2),
            )
        except ImportError:
            return PerceptionResult(
                input_type="document", source=path,
                description="", extracted={}, raw_text="",
                confidence=0.0, elapsed=time.time()-t0,
                error="Install: pip install python-docx",
            )

    def _perceive_text(self, path, question, max_chars, t0) -> PerceptionResult:
        try:
            with open(path, encoding="utf-8", errors="ignore") as f:
                text = f.read()[:max_chars]
            return PerceptionResult(
                input_type="document", source=path,
                description=f"Text document, {len(text)} chars",
                extracted=self._extract_document_structure(text),
                raw_text=text,
                confidence=0.95,
                elapsed=round(time.time()-t0, 2),
            )
        except Exception as e:
            return PerceptionResult(
                input_type="document", source=path,
                description="", extracted={}, raw_text="",
                confidence=0.0, elapsed=time.time()-t0,
                error=str(e)[:100],
            )

    @staticmethod
    def _extract_document_structure(text: str) -> dict:
        extracted = {}
        lines = text.split("\n")
        headings = [l.strip() for l in lines
                    if l.strip() and (l.startswith("#") or l.isupper())]
        if headings:
            extracted["sections"] = headings[:5]
        equations = re.findall(r"\$[^$]+\$|\\begin\{equation\}|[A-Za-z]\s*=\s*[^,\n]{5,}", text)
        if equations:
            extracted["mathematical_content"] = equations[:3]
        citations = re.findall(r"\[\d+\]|\([A-Z][a-z]+,?\s+\d{4}\)", text)
        if citations:
            extracted["citation_count"] = len(citations)
        return extracted


# ============================================================
# PerceptionHub
# ============================================================

IMAGE_EXTENSIONS  = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tiff"}
AUDIO_EXTENSIONS  = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac"}
DOC_EXTENSIONS    = {".pdf", ".docx", ".doc", ".txt", ".md", ".rst"}


class PerceptionHub:
    """
    Coordinates all perceivers. Auto-detects input type.
    Formats results for injection into the reasoning chain.
    """

    def __init__(
        self,
        api_key:         str  = "",
        whisper_model:   str  = "base",
        verbose:         bool = True,
    ):
        self.image    = ImagePerceiver(api_key=api_key, verbose=verbose)
        self.audio    = AudioPerceiver(model_size=whisper_model, verbose=verbose)
        self.document = DocumentPerceiver()
        self.verbose  = verbose
        self._history: list[dict] = []

    def perceive(
        self,
        source:   str,
        question: str = "",
    ) -> PerceptionResult:
        """
        Perceive any input and return a structured result.
        Auto-detects type from extension or URL pattern.
        """
        ext = Path(source).suffix.lower()

        if ext in IMAGE_EXTENSIONS or self._looks_like_image_url(source):
            result = self.image.perceive(source, question)
        elif ext in AUDIO_EXTENSIONS:
            result = self.audio.perceive(source, question)
        elif ext in DOC_EXTENSIONS:
            result = self.document.perceive(source, question)
        else:
            result = PerceptionResult(
                input_type="unknown", source=source,
                description=f"Unknown input type: {ext}",
                extracted={}, raw_text="",
                confidence=0.0,
                error=f"Cannot perceive {ext} files",
            )

        self._history.append(result.to_dict())
        return result

    def perceive_multiple(
        self,
        sources:  list[str],
        question: str = "",
    ) -> list[PerceptionResult]:
        """Perceive multiple inputs and return all results."""
        return [self.perceive(s, question) for s in sources]

    def format_all_for_reasoning(
        self,
        results: list[PerceptionResult],
        question: str = "",
    ) -> str:
        """Format multiple perception results as one context block."""
        formatted = [r.format_for_reasoning(question) for r in results
                     if not r.error]
        return "\n\n".join(formatted) if formatted else ""

    @staticmethod
    def _looks_like_image_url(url: str) -> bool:
        return any(url.lower().endswith(ext) for ext in IMAGE_EXTENSIONS)

    def status(self) -> str:
        n = len(self._history)
        types = {}
        for h in self._history:
            t = h.get("input_type", "?")
            types[t] = types.get(t, 0) + 1
        return (
            f"PerceptionHub: {n} items perceived — "
            + ", ".join(f"{t}:{c}" for t, c in types.items())
        )
