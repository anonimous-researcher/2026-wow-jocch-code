
import time
import logging
import torch
import transformers
import liqfit.models
import liqfit.pipeline
import pandas


#
#
#
#
#
####################### NLI MODELS CODE #######################
#
#
#
#
#

def keywords_extraction(df: pandas.DataFrame,
                        model_id: str, verbose: bool,
                        batch_size: int = 8,
                        label_template: str = "{}") -> pandas.DataFrame:

    # Set up logger
    logger = logging.getLogger("KeywordExtraction")
    logger.setLevel(logging.DEBUG if verbose else logging.WARNING)

    logger.info(f"Using model: {model_id}")

    # Device selection
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    logger.info(f"Selected device: {device}")

    # Extract texts and class labels
    texts = df["text"].tolist()
    raw_labels = df.columns[1:].tolist()
    candidate_labels = [label_template.format(label) for label in raw_labels]

    # Load model & pipeline
    if model_id == "knowledgator/comprehend_it-multilingual-t5-base":
        model = liqfit.models.T5ForZeroShotClassification.from_pretrained(model_id).to(device)
        tokenizer = transformers.T5Tokenizer.from_pretrained(model_id)
        classifier = liqfit.pipeline.ZeroShotClassificationPipeline(
            model=model,
            tokenizer=tokenizer,
            hypothesis_template=label_template,
            encoder_decoder=True,
            device=device
        )
    else:
        classifier = transformers.pipeline(
            "zero-shot-classification",
            model=model_id,
            tokenizer=transformers.AutoTokenizer.from_pretrained(model_id),
            device=0 if device == "cuda" else -1
        )

    # Initialize result storage
    results_dict = {
        "text": [],
        "time": []
    }
    for label in raw_labels:
        results_dict[label] = []

    # Process in batches
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        if verbose:
            logger.info(f"Processing batch {i}–{i + len(batch_texts) - 1}")

        start_time = time.perf_counter()
        outputs = classifier(batch_texts, candidate_labels=candidate_labels, multi_label=True)
        elapsed_time = time.perf_counter() - start_time

        if isinstance(outputs, dict):  # Happens when batch size = 1
            outputs = [outputs]

        for text, output in zip(batch_texts, outputs):
            results_dict["text"].append(text)
            results_dict["time"].append(elapsed_time / len(batch_texts))

            score_map = {label: 0.0 for label in candidate_labels}
            for label, score in zip(output["labels"], output["scores"]):
                score_map[label] = score

            for raw_label, templated_label in zip(raw_labels, candidate_labels):
                results_dict[raw_label].append(score_map[templated_label])

    return pandas.DataFrame(results_dict)

#
#
#
#
#
####################### DECODER-ONLY MODELS CODE #######################
#
#
#
#
#


def _select_device(logger: logging.Logger) -> str:
    """
    Select the best available device in the following priority:
    CUDA > MPS (Apple Silicon) > CPU.
    """
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    logger.info(f"Selected device: {device}")
    return device


def _ensure_padding_token(tokenizer: transformers.PreTrainedTokenizerBase) -> None:
    """
    Many decoder-only tokenizers do not define a pad_token.
    For batched inference with padding, we set pad_token = eos_token when needed.
    """
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


def _pick_single_token_choice_pair(
    tokenizer: transformers.PreTrainedTokenizerBase
) -> tuple[str, str, int, int]:
    """
    Choose a robust positive/negative pair that is encoded as a SINGLE token each.
    We use the next-token logits at the end of the prompt to compute P(pos).

    Different tokenizers split differently; we try a small set of candidates.
    Returns:
        pos_str, neg_str, pos_token_id, neg_token_id
    """
    candidate_pairs = [
        (" Yes", " No"),
        (" true", " false"),
        (" True", " False"),
        (" 1", " 0"),
        (" A", " B"),
    ]

    for pos_str, neg_str in candidate_pairs:
        pos_ids = tokenizer.encode(pos_str, add_special_tokens=False)
        neg_ids = tokenizer.encode(neg_str, add_special_tokens=False)

        if len(pos_ids) == 1 and len(neg_ids) == 1:
            return pos_str, neg_str, pos_ids[0], neg_ids[0]

    raise ValueError(
        "No suitable single-token (positive, negative) choice pair found for this tokenizer. "
        "To keep the setup simple and deterministic, add more candidate pairs in "
        "_pick_single_token_choice_pair()."
    )


def _pick_choice_pair(tokenizer: transformers.PreTrainedTokenizerBase):
    """
    Choose a positive/negative answer pair for log-prob scoring.
    We allow multi-token strings and return their token-id sequences.
    """

    # Try several common pairs; we will accept the first that produces non-empty token sequences.
    candidate_pairs = [
        (" Yes", " No"),
        (" true", " false"),
        (" True", " False"),
        (" 1", " 0"),
        (" A", " B"),
        (" si", " no"),     # extra for IT (may still tokenize fine)
        (" sì", " no"),     # note accent; tokenizer-dependent
    ]

    for pos_str, neg_str in candidate_pairs:
        pos_ids = tokenizer.encode(pos_str, add_special_tokens=False)
        neg_ids = tokenizer.encode(neg_str, add_special_tokens=False)

        if len(pos_ids) > 0 and len(neg_ids) > 0:
            return pos_str, neg_str, pos_ids, neg_ids

    raise ValueError(
        "No suitable (positive, negative) choice pair found for this tokenizer. "
        "Please add more candidates in _pick_choice_pair()."
    )

def _build_entailment_prompt(text: str, templated_label: str, prompt_language: str) -> str:
    """
    Build a minimal, fixed entailment-style prompt.
    We keep the structure identical across languages, translating only the scaffolding.
    """
    language = (prompt_language or "EN").upper()

    if language == "IT":
        return (
            "Testo:\n"
            f"{text}\n\n"
            "Ipotesi:\n"
            f"{templated_label}\n\n"
            "Domanda: L'ipotesi e' implicata (entailed) dal testo?\n"
            "Risposta:"
        )

    # Default: English
    return (
        "Text:\n"
        f"{text}\n\n"
        "Hypothesis:\n"
        f"{templated_label}\n\n"
        "Question: Is the hypothesis entailed by the text?\n"
        "Answer:"
    )


@torch.no_grad()
def _score_prompts_next_token_probability(
    model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizerBase,
    prompts: list[str],
    pos_token_id: int,
    neg_token_id: int,
    max_length: int
) -> list[float]:
    """
    Compute P(pos | prompt) by looking at the NEXT token distribution right after the prompt.
    We only compare the logits of pos_token and neg_token for stability and simplicity:
        p = sigmoid(logit_pos - logit_neg)
    """
    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length
    )

    input_ids = encoded["input_ids"].to(model.device)
    attention_mask = encoded["attention_mask"].to(model.device)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits  # [batch, seq_len, vocab]

    # Identify the last non-pad position for each prompt
    last_positions = attention_mask.sum(dim=1) - 1  # [batch]
    batch_indices = torch.arange(input_ids.size(0), device=logits.device)

    last_logits = logits[batch_indices, last_positions, :]  # [batch, vocab]

    pos_logits = last_logits[:, pos_token_id]
    neg_logits = last_logits[:, neg_token_id]

    probabilities = torch.sigmoid(pos_logits - neg_logits)
    return probabilities.detach().cpu().tolist()


@torch.no_grad()
def _score_prompts_choice_sequence_probability(
    model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizerBase,
    prompts: list[str],
    pos_token_ids: list[int],
    neg_token_ids: list[int],
    max_length: int
) -> list[float]:
    """
    Compute P(pos | prompt) using sequence log-probabilities for pos/neg token sequences.

    For each prompt:
      score_pos = sum_t log P(pos_t | prompt + pos_<t)
      score_neg = sum_t log P(neg_t | prompt + neg_<t)

      p = sigmoid(score_pos - score_neg)

    This works even if the answer strings are tokenized into multiple tokens.
    """

    # Tokenize prompts once
    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length
    )

    input_ids = encoded["input_ids"].to(model.device)
    attention_mask = encoded["attention_mask"].to(model.device)

    # We will compute scores in a loop over the answer tokens.
    # This is still deterministic and usually fast enough for short answers (1–4 tokens).
    batch_size = input_ids.size(0)

    def sequence_logprob(answer_ids: list[int]) -> torch.Tensor:
        """
        Compute total log-prob of generating answer_ids after each prompt in the batch.
        Returns: tensor shape [batch_size]
        """
        # Start with prompt
        current_input_ids = input_ids
        current_attention_mask = attention_mask

        total_logprob = torch.zeros(batch_size, device=model.device)

        for token_id in answer_ids:
            outputs = model(input_ids=current_input_ids, attention_mask=current_attention_mask)
            logits = outputs.logits  # [B, T, V]

            # last non-pad position for each row
            last_positions = current_attention_mask.sum(dim=1) - 1
            batch_indices = torch.arange(batch_size, device=model.device)
            last_logits = logits[batch_indices, last_positions, :]  # [B, V]

            log_probs = torch.log_softmax(last_logits, dim=-1)
            total_logprob += log_probs[:, token_id]

            # append generated token to continue the conditional chain
            token_tensor = torch.full((batch_size, 1), token_id, dtype=current_input_ids.dtype, device=model.device)
            current_input_ids = torch.cat([current_input_ids, token_tensor], dim=1)

            # extend attention mask with 1s
            ones_mask = torch.ones((batch_size, 1), dtype=current_attention_mask.dtype, device=model.device)
            current_attention_mask = torch.cat([current_attention_mask, ones_mask], dim=1)

        return total_logprob

    pos_lp = sequence_logprob(pos_token_ids)
    neg_lp = sequence_logprob(neg_token_ids)

    probs = torch.sigmoid(pos_lp - neg_lp)
    return probs.detach().cpu().tolist()


def decoder_only_keywords_extraction(
    df: pandas.DataFrame,
    model_id: str,
    verbose: bool,
    batch_size: int = 8,
    label_template: str = "{}",
    label_chunk_size: int = 32,
    max_length: int = 1024,
    prompt_language: str = "EN",
    prompt_batch_size: int = 256
) -> pandas.DataFrame:
    """
    Decoder-only analogue of keywords_extraction().

    It computes a score in [0, 1] for each (text, label) pair using a fixed YES/NO decision
    via log-prob scoring of positive/negative answer token sequences.

    H100-friendly improvements:
      - BF16 on CUDA
      - FlashAttention-2 when available (fallback if not)
      - prompt micro-batching via prompt_batch_size
      - left padding for decoder-only batching

    Parameters (new):
      - prompt_batch_size: micro-batch size for the *prompt list* inside each label chunk.
        On H100 you can try 512 or 1024. Keep smaller if prompts are long.

    Returns:
      DataFrame with:
        - "text"
        - "time" (average inference time per text within the batch)
        - one column per raw label containing scores in [0, 1]
    """
    logger = logging.getLogger("DecoderOnlyKeywordExtraction")
    logger.setLevel(logging.DEBUG if verbose else logging.WARNING)
    logger.info(f"Using decoder-only model: {model_id}")

    device = _select_device(logger)

    texts = df["text"].tolist()
    raw_labels = df.columns[1:].tolist()
    candidate_labels = [label_template.format(label) for label in raw_labels]

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, use_fast=True)
    _ensure_padding_token(tokenizer)

    # Decoder-only batching typically prefers left padding.
    tokenizer.padding_side = "left"

    # Load model with device-appropriate dtype and (on CUDA) attention backend.
    if device == "cuda":
        # Good default for Ampere/Hopper.
        torch.backends.cuda.matmul.allow_tf32 = True

        # Try FlashAttention-2; if it fails, fallback to default attention.
        try:
            model = transformers.AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                attn_implementation="flash_attention_2"
            )
            logger.info("Loaded model with BF16 + FlashAttention-2.")
        except Exception as exc:
            logger.warning(f"FlashAttention-2 not available for this setup/model. Falling back. Reason: {exc}")
            model = transformers.AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )

    elif device == "mps":
        # Conservative settings for Apple Silicon to avoid memory spikes
        if batch_size > 1:
            batch_size = 1
        if label_chunk_size > 8:
            label_chunk_size = 8
        if max_length > 384:
            max_length = 384
        if prompt_batch_size > 8:
            prompt_batch_size = 8

        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16
        )
        model = model.to(device)

    else:
        model = transformers.AutoModelForCausalLM.from_pretrained(model_id)
        model = model.to(device)

    model.eval()

    pos_str, neg_str, pos_token_ids, neg_token_ids = _pick_choice_pair(tokenizer)
    logger.info(f"Choice pair: '{pos_str.strip()}' vs '{neg_str.strip()}'")

    results_dict: dict[str, list] = {"text": [], "time": []}
    for label in raw_labels:
        results_dict[label] = []

    # Outer loop over text batches
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        if verbose:
            logger.info(f"Processing text batch {i}–{i + len(batch_texts) - 1}")

        batch_start_time = time.perf_counter()

        # Prepare storage for this batch: scores[text_index][label_index]
        batch_scores = [
            [0.0 for _ in range(len(raw_labels))]
            for _ in range(len(batch_texts))
        ]

        # Inner loop: chunk labels to keep prompt lists bounded
        for j in range(0, len(candidate_labels), label_chunk_size):
            label_chunk = candidate_labels[j:j + label_chunk_size]

            prompts: list[str] = []
            for text in batch_texts:
                for templated_label in label_chunk:
                    prompts.append(_build_entailment_prompt(text, templated_label, prompt_language))

            # Micro-batch the prompt list to exploit GPU efficiently and avoid large allocations.
            chunk_probabilities: list[float] = []
            for k in range(0, len(prompts), prompt_batch_size):
                chunk_probabilities.extend(
                    _score_prompts_choice_sequence_probability(
                        model=model,
                        tokenizer=tokenizer,
                        prompts=prompts[k:k + prompt_batch_size],
                        pos_token_ids=pos_token_ids,
                        neg_token_ids=neg_token_ids,
                        max_length=max_length
                    )
                )

            # Map back chunk results into the batch_scores matrix
            chunk_len = len(label_chunk)
            idx = 0
            for text_index in range(len(batch_texts)):
                for chunk_label_index in range(chunk_len):
                    global_label_index = j + chunk_label_index
                    batch_scores[text_index][global_label_index] = chunk_probabilities[idx]
                    idx += 1

        elapsed = time.perf_counter() - batch_start_time
        avg_time_per_text = elapsed / max(1, len(batch_texts))

        for text_index, text in enumerate(batch_texts):
            results_dict["text"].append(text)
            results_dict["time"].append(avg_time_per_text)
            for raw_label, score in zip(raw_labels, batch_scores[text_index]):
                results_dict[raw_label].append(score)

    return pandas.DataFrame(results_dict)
