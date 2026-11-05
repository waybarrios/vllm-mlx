# SPDX-License-Identifier: Apache-2.0
"""Counting the tokens of a rendered prompt the way the engines encode it."""

from vllm_mlx.utils.tokenizer import count_prompt_tokens


class _Tokenizer:
    """One token per word; ``encode`` prepends BOS when asked to add specials."""

    def __init__(self, **attrs):
        self.calls = []
        self.__dict__.update(attrs)

    def encode(self, text, add_special_tokens=True):
        self.calls.append(add_special_tokens)
        words = text.split()
        return ["<s>"] + words if add_special_tokens else words


def test_prompt_without_bos_is_encoded_with_special_tokens():
    tokenizer = _Tokenizer(bos_token="<s>")

    assert count_prompt_tokens(tokenizer, "a b c") == 4
    assert tokenizer.calls == [True]


def test_prompt_starting_with_bos_is_encoded_without_special_tokens():
    tokenizer = _Tokenizer(bos_token="<s>")

    assert count_prompt_tokens(tokenizer, "<s> a b c") == 4
    assert tokenizer.calls == [False]


def test_bos_in_the_middle_of_a_prompt_does_not_count_as_leading():
    tokenizer = _Tokenizer(bos_token="<s>")

    assert count_prompt_tokens(tokenizer, "a <s> b") == 4
    assert tokenizer.calls == [True]


def test_tokenizer_without_a_bos_token_adds_special_tokens():
    for tokenizer in (_Tokenizer(bos_token=None), _Tokenizer()):
        assert count_prompt_tokens(tokenizer, "a b") == 3
        assert tokenizer.calls == [True]


def test_empty_prompt_counts_the_special_tokens_only():
    assert count_prompt_tokens(_Tokenizer(bos_token="<s>"), "") == 1
