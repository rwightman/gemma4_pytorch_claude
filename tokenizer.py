"""Gemma4 tokenizer — thin SentencePiece wrapper.

Matches the JAX ``gm.text._tokenizer.Gemma4Tokenizer`` API.
"""

from __future__ import annotations

import sentencepiece as spm


class Gemma4Tokenizer:
    """Lightweight SentencePiece wrapper for Gemma 4.

    Usage::

        tok = Gemma4Tokenizer("/path/to/gemma4.model")
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
        self._sp = spm.SentencePieceProcessor()
        self._sp.Load(model_path)

    def encode(
        self,
        text: str,
        *,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[int]:
        """Encode text to token IDs."""
        ids: list[int] = self._sp.EncodeAsIds(text)
        if add_bos:
            ids.insert(0, self.BOS)
        if add_eos:
            ids.append(self.EOS)
        return ids

    def decode(self, ids: list[int] | int) -> str:
        """Decode token IDs back to text."""
        if isinstance(ids, int):
            ids = [ids]
        return self._sp.DecodeIds(ids)

    @property
    def vocab_size(self) -> int:
        return self._sp.GetPieceSize()
