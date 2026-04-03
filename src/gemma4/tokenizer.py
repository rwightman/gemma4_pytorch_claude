"""Gemma4 tokenizer — supports SentencePiece ``.model`` and HuggingFace ``tokenizer.json``.

Matches the JAX ``gm.text._tokenizer.Gemma4Tokenizer`` API.
"""

from __future__ import annotations

from pathlib import Path


class Gemma4Tokenizer:
    """Lightweight tokenizer for Gemma 4.

    Supports two backends:

    * **SentencePiece**: pass a path to a ``.model`` file.
    * **HuggingFace tokenizers**: pass a path to a ``tokenizer.json`` file
      or a directory containing one.

    Usage::

        tok = Gemma4Tokenizer("/path/to/gemma4.model")
        # or
        tok = Gemma4Tokenizer("/path/to/hf-model-dir/")

        ids = tok.encode("Hello!", add_bos=True)
        text = tok.decode(ids)
    """

    # Special token IDs (matching _Gemma4SpecialTokens in JAX reference)
    PAD = 0
    EOS = 1
    BOS = 2
    UNK = 3
    MASK = 4
    START_OF_TURN = 105
    END_OF_TURN = 106
    IMAGE_PLACEHOLDER = 258880
    START_OF_IMAGE = 255999
    END_OF_IMAGE = 258882
    AUDIO_PLACEHOLDER = 258881
    START_OF_AUDIO = 256000
    END_OF_AUDIO = 258883
    BEGIN_OF_TOOL_RESPONSE = 50

    def __init__(self, model_path: str) -> None:
        p = Path(model_path)

        # Determine backend
        if p.is_dir():
            json_path = p / "tokenizer.json"
            model_path_sp = p / "tokenizer.model"
            if json_path.exists():
                self._backend = "hf"
                self._init_hf(str(json_path))
            elif model_path_sp.exists():
                self._backend = "sp"
                self._init_sp(str(model_path_sp))
            else:
                raise FileNotFoundError(
                    f"No tokenizer.json or tokenizer.model found in {p}"
                )
        elif p.suffix == ".json":
            self._backend = "hf"
            self._init_hf(str(p))
        elif p.suffix == ".model":
            self._backend = "sp"
            self._init_sp(str(p))
        else:
            raise ValueError(
                f"Unknown tokenizer file type: {p.suffix}. "
                "Expected .model (SentencePiece) or .json (HuggingFace)."
            )

    def _init_sp(self, path: str) -> None:
        import sentencepiece as spm
        self._sp = spm.SentencePieceProcessor()
        self._sp.Load(path)

    def _init_hf(self, path: str) -> None:
        from tokenizers import Tokenizer
        self._hf_tok = Tokenizer.from_file(path)

    def encode(
            self,
            text: str,
            *,
            add_bos: bool = False,
            add_eos: bool = False,
    ) -> list[int]:
        """Encode text to token IDs."""
        if self._backend == "sp":
            ids: list[int] = self._sp.EncodeAsIds(text)
        else:
            encoding = self._hf_tok.encode(text, add_special_tokens=False)
            ids = encoding.ids
        if add_bos:
            ids.insert(0, self.BOS)
        if add_eos:
            ids.append(self.EOS)
        return ids

    def decode(self, ids: list[int] | int) -> str:
        """Decode token IDs back to text."""
        if isinstance(ids, int):
            ids = [ids]
        if self._backend == "sp":
            return self._sp.DecodeIds(ids)
        else:
            return self._hf_tok.decode(ids, skip_special_tokens=False)

    @property
    def vocab_size(self) -> int:
        if self._backend == "sp":
            return self._sp.GetPieceSize()
        else:
            return self._hf_tok.get_vocab_size()
