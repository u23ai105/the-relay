"""
Phase 9 / Batch 1: 18 Independently Authored Evaluation Questions and Deterministic Rubric.
Covers:
- Answerable single-fact, equation, table, and hyperparameter questions across all 3 papers.
- Multi-trap / naive-retriever trap questions (where shared terms like 'attention',
  'tokens', 'chunking', or 'streaming' appear across multiple papers).
- Unanswerable questions (both subtle in-domain traps asking for experiments the papers
  did not perform and out-of-domain questions).
Separates infrastructure failures (status != 'ok') from answer-quality grades.
"""

from dataclasses import dataclass, field
import re
from typing import Any, Dict, List, Optional

from src.prompts_and_schemas import GenerationOutput


@dataclass
class EvalQuestion:
    question_id: str
    question: str
    category: str  # "answerable", "naive_trap_answerable", "unanswerable_in_domain", "unanswerable_out_of_domain"
    expected_abstained: bool
    expected_paper_id: Optional[str]
    expected_arxiv_version: Optional[str]
    expected_pages: List[int] = field(default_factory=list)
    required_any_keywords: List[str] = field(default_factory=list)
    required_all_keywords: List[str] = field(default_factory=list)
    forbidden_keywords: List[str] = field(default_factory=list)
    reference_answer: str = ""
    naive_retriever_trap_note: str = ""


EVALUATION_QUESTIONS: List[EvalQuestion] = [
    # --- Answerable Questions on Attention Is All You Need (1706.03762v7) ---
    EvalQuestion(
        question_id="q01",
        question="In Attention Is All You Need (1706.03762v7), how many parallel attention layers (heads) h are used in the base Transformer model, and what are the query/key dimension d_k and model dimension d_model?",
        category="answerable",
        expected_abstained=False,
        expected_paper_id="1706.03762",
        expected_arxiv_version="v7",
        expected_pages=[4, 5, 9],
        required_all_keywords=["8", "64", "512"],
        reference_answer="The base Transformer uses h = 8 parallel attention layers (heads), d_k = d_v = 64, and d_model = 512 (Section 3.2.2, p. 5; Table 3, p. 9).",
    ),
    EvalQuestion(
        question_id="q02",
        question="What BLEU scores did the big Transformer model achieve on the WMT 2014 English-to-German and English-to-French translation tasks in Attention Is All You Need?",
        category="answerable",
        expected_abstained=False,
        expected_paper_id="1706.03762",
        expected_arxiv_version="v7",
        expected_pages=[1, 2, 8],
        required_all_keywords=["28.4", "41.8"],
        reference_answer="The big Transformer model achieved 28.4 BLEU on WMT 2014 English-to-German and 41.8 BLEU on WMT 2014 English-to-French (Abstract, p. 1; Section 6.1, Table 2, p. 8).",
    ),
    EvalQuestion(
        question_id="q03",
        question="What Adam optimizer hyperparameters (beta_1, beta_2, epsilon) and warmup_steps value were used to train the Transformer in Attention Is All You Need?",
        category="answerable",
        expected_abstained=False,
        expected_paper_id="1706.03762",
        expected_arxiv_version="v7",
        expected_pages=[7, 8],
        required_all_keywords=["0.9", "0.98", "4000"],
        reference_answer="They used the Adam optimizer with beta_1 = 0.9, beta_2 = 0.98, epsilon = 10^-9, and warmup_steps = 4000 (Section 5.3 Optimizer, p. 7).",
    ),
    EvalQuestion(
        question_id="q04",
        question="What value of label smoothing (epsilon_ls) was used during training in Attention Is All You Need, and how did it affect perplexity and BLEU score?",
        category="answerable",
        expected_abstained=False,
        expected_paper_id="1706.03762",
        expected_arxiv_version="v7",
        expected_pages=[8],
        required_all_keywords=["0.1", "perplexity"],
        reference_answer="Label smoothing of value epsilon_ls = 0.1 was applied; it hurts perplexity as the model learns to be more unsure, but improves accuracy and BLEU score (Section 5.4 Regularization, p. 8).",
    ),
    # --- Answerable Questions on Moshi (2410.00037v1) ---
    EvalQuestion(
        question_id="q05",
        question="In the Moshi paper (2410.00037v1), what is the name of the text language model backbone, how many parameters does it have, and how many hours of unsupervised audio data were used in audio pre-training?",
        category="answerable",
        expected_abstained=False,
        expected_paper_id="2410.00037",
        expected_arxiv_version="v1",
        expected_pages=[1, 2, 3, 6, 7, 17],
        required_all_keywords=["helium", "7b"],
        required_any_keywords=["7 million", "7m"],
        reference_answer="The text backbone is Helium, a 7B-parameter language model, and audio pre-training used 7 million hours of unsupervised audio content (Section 1, p. 2-3; Section 3.2, p. 7; Section 4.2, p. 17).",
    ),
    EvalQuestion(
        question_id="q06",
        question="What frame rate (in Hz) and bitrate (in kbps) does the Mimi neural audio codec operate at in Moshi (2410.00037v1), and what quantization architecture does it use?",
        category="naive_trap_answerable",
        expected_abstained=False,
        expected_paper_id="2410.00037",
        expected_arxiv_version="v1",
        expected_pages=[2, 3, 9, 10, 11],
        required_all_keywords=["12.5", "1.1"],
        required_any_keywords=["residual vector quantization", "rvq", "split"],
        reference_answer="Mimi operates at a frame rate of 12.5 Hz and a bitrate of 1.1 kbps (with 8 quantizers) using a split residual vector quantization (RVQ) bottleneck (Section 1, p. 2; Section 3.3 Audio Tokenization, p. 9-11).",
        naive_retriever_trap_note="Trap 1: Queries about 'audio tokenization rate and quantization' collide with Dynamic Chunking's 'tokenization compression ratio' unless routed/ranked properly.",
    ),
    EvalQuestion(
        question_id="q07",
        question="What is Inner Monologue in Moshi (2410.00037v1), and what is Moshi's theoretical latency versus achieved practical latency?",
        category="answerable",
        expected_abstained=False,
        expected_paper_id="2410.00037",
        expected_arxiv_version="v1",
        expected_pages=[1, 2, 3, 6, 12, 13, 14],
        required_all_keywords=["160", "200"],
        required_any_keywords=["text tokens", "time-aligned", "prefix", "whisper"],
        reference_answer="Inner Monologue predicts time-aligned text tokens as a prefix to audio tokens, improving linguistic quality and enabling streaming ASR/TTS, with a theoretical latency of 160 ms and practical latency of 200 ms (Abstract, p. 1; Section 1, p. 2; Section 3.4, p. 12-14).",
    ),
    EvalQuestion(
        question_id="q08",
        question="How does the RQ-Transformer in Moshi decompose sequence modeling into a Temporal Transformer and a Depth Transformer?",
        category="naive_trap_answerable",
        expected_abstained=False,
        expected_paper_id="2410.00037",
        expected_arxiv_version="v1",
        expected_pages=[2, 3, 12, 13],
        required_any_keywords=["temporal", "depth", "sub-sequence", "steps"],
        reference_answer="The RQ-Transformer decomposes the flattened sequence of K residual audio codebooks across T timesteps into a large Temporal Transformer operating over the T timesteps and a smaller Depth Transformer predicting the K codebook tokens at each timestep (Section 3.4, p. 12-13, Figure 3).",
        naive_retriever_trap_note="Trap 2: Queries about 'hierarchical sequence modeling across timesteps' strongly match Dynamic Chunking (2507.07955v2) unless RQ-Transformer / Temporal+Depth specifics are preserved.",
    ),
    # --- Answerable Questions on Dynamic Chunking (2507.07955v2) ---
    EvalQuestion(
        question_id="q09",
        question="In Dynamic Chunking for End-to-End Hierarchical Sequence Modeling (2507.07955v2), what are the two auxiliary mechanisms inside Dynamic Chunking (DC) that enable end-to-end gradient-based learning of discrete chunk boundaries?",
        category="answerable",
        expected_abstained=False,
        expected_paper_id="2507.07955",
        expected_arxiv_version="v2",
        expected_pages=[1, 2, 3, 6, 7],
        required_all_keywords=["routing", "smoothing"],
        required_any_keywords=["cosine similarity", "boundary", "ratio loss"],
        reference_answer="Dynamic Chunking uses (1) a routing module that predicts boundary probabilities between adjacent elements via scaled cosine similarity, and (2) a smoothing module (plus a ratio loss) that interpolates representations during dechunking to make discrete boundary selection differentiable end-to-end (Section 2.2, p. 6-7).",
    ),
    EvalQuestion(
        question_id="q10",
        question="In the H-Net architecture (2507.07955v2), why does the encoder and decoder use Mamba-2 (SSM) layers while the main network uses Transformer layers?",
        category="naive_trap_answerable",
        expected_abstained=False,
        expected_paper_id="2507.07955",
        expected_arxiv_version="v2",
        expected_pages=[4, 5, 11, 18],
        required_any_keywords=["mamba", "ssm", "byte", "fine-grained", "compression", "inductive bias"],
        reference_answer="Mamba-2 (SSM) layers have stronger inductive biases and efficiency for fine-grained raw byte sequences in the outer encoder/decoder stages, while Transformer layers excel in the compressed main network once sequences are dynamically chunked into higher-level representations (Section 2.1, p. 4-5; Section 3.3, p. 18).",
        naive_retriever_trap_note="Trap 3: Mentioning 'encoder and decoder stacks and Transformer layers' pulls chunks from Attention Is All You Need Section 3.1 unless H-Net / Mamba-2 context is matched.",
    ),
    EvalQuestion(
        question_id="q11",
        question="What is the role of the Ratio Loss in Dynamic Chunking (2507.07955v2), and what happens during training without a target compression ratio constraint?",
        category="answerable",
        expected_abstained=False,
        expected_paper_id="2507.07955",
        expected_arxiv_version="v2",
        expected_pages=[6, 7, 8],
        required_all_keywords=["ratio", "loss"],
        required_any_keywords=["collapse", "compression", "boundary", "trivial", "all tokens"],
        reference_answer="The Ratio Loss regularizes the fraction of selected chunk boundaries toward a target compression ratio N, preventing the routing module from collapsing into the trivial solution of selecting every token as a boundary (no compression) (Section 2.2.3 Ratio Loss, p. 7-8).",
    ),
    EvalQuestion(
        question_id="q12",
        question="On which non-English natural language and biological sequence datasets was H-Net evaluated in Dynamic Chunking (2507.07955v2 Section 3.2)?",
        category="answerable",
        expected_abstained=False,
        expected_paper_id="2507.07955",
        expected_arxiv_version="v2",
        expected_pages=[2, 10, 14, 15, 16],
        required_all_keywords=["chinese", "dna"],
        reference_answer="H-Net was evaluated on Chinese language modeling, code modeling, and DNA sequence datasets (Section 3.2 Alternate Language Datasets, p. 14-16).",
    ),
    # --- Unanswerable In-Domain Trap Questions (Must Refuse) ---
    EvalQuestion(
        question_id="q13",
        question="What BLEU score did the Transformer achieve on the WMT 2014 English-to-Japanese translation benchmark in Attention Is All You Need (1706.03762v7)?",
        category="unanswerable_in_domain",
        expected_abstained=True,
        expected_paper_id=None,
        expected_arxiv_version=None,
        reference_answer="Refusal required: Attention Is All You Need only reports WMT 2014 translation results on English-to-German and English-to-French, not English-to-Japanese.",
    ),
    EvalQuestion(
        question_id="q14",
        question="In Moshi (2410.00037v1), how many video frames per second does the visual encoder process during multimodal video-and-speech dialogue?",
        category="unanswerable_in_domain",
        expected_abstained=True,
        expected_paper_id=None,
        expected_arxiv_version=None,
        reference_answer="Refusal required: Moshi is a speech-text foundation model and does not include a visual or video frame encoder.",
    ),
    EvalQuestion(
        question_id="q15",
        question="What top-1 classification accuracy did H-Net achieve on the ImageNet-1K 2D image classification benchmark in Dynamic Chunking (2507.07955v2)?",
        category="unanswerable_in_domain",
        expected_abstained=True,
        expected_paper_id=None,
        expected_arxiv_version=None,
        reference_answer="Refusal required: Dynamic Chunking (2507.07955v2) evaluates autoregressive sequence modeling on text, Chinese, code, and DNA, not ImageNet-1K image classification.",
    ),
    EvalQuestion(
        question_id="q16",
        question="In Attention Is All You Need (1706.03762v7), what was the exact electricity bill in USD paid to the cloud provider for training the base model on 8 P100 GPUs?",
        category="unanswerable_in_domain",
        expected_abstained=True,
        expected_paper_id=None,
        expected_arxiv_version=None,
        reference_answer="Refusal required: The paper reports training duration (12 hours for base, 3.5 days for big on 8 P100 GPUs), but does not report the electricity bill or cloud dollar cost.",
    ),
    # --- Unanswerable Out-of-Domain Questions (Must Refuse) ---
    EvalQuestion(
        question_id="q17",
        question="How does the Raft consensus algorithm handle leader election when a network partition splits a 5-node cluster?",
        category="unanswerable_out_of_domain",
        expected_abstained=True,
        expected_paper_id=None,
        expected_arxiv_version=None,
        reference_answer="Refusal required: Raft consensus and distributed leader election are not covered in any of the three papers.",
    ),
    EvalQuestion(
        question_id="q18",
        question="What is the chemical synthesis procedure for preparing lithium iron phosphate (LiFePO4) cathode active material?",
        category="unanswerable_out_of_domain",
        expected_abstained=True,
        expected_paper_id=None,
        expected_arxiv_version=None,
        reference_answer="Refusal required: Battery cathode chemistry is outside the three-paper research corpus.",
    ),
]


def get_question_by_id(question_id: str) -> EvalQuestion:
    for q in EVALUATION_QUESTIONS:
        if q.question_id == question_id:
            return q
    raise KeyError(f"Unknown question_id: {question_id}")


def grade_pipeline_response(
    eval_q: EvalQuestion,
    status: str,
    gen_output: Optional[GenerationOutput],
    citation_valid: bool,
) -> str:
    """
    Grades a pipeline execution strictly into the required `answer_grade` enum:
      - "" (empty string) if `status != "ok"` (ungraded infrastructure failure; never
        conflated with wrong answers or refusals).
      - "refused_correctly" if the question is unanswerable (`expected_abstained=True`)
        and the assistant abstained (`gen_output.abstained=True`).
      - "refused_wrongly" if the question is answerable (`expected_abstained=False`)
        and the assistant abstained (`gen_output.abstained=True`).
      - "correct" if the question is answerable, `abstained=False`, `citation_valid=True`,
        cites the expected paper_id and arxiv_version, and satisfies the keyword rubric.
      - "wrong" otherwise (e.g., answered an unanswerable question, failed citation
        validity, cited the wrong paper, or stated incorrect facts).
    """
    if status != "ok" or gen_output is None:
        return ""

    if eval_q.expected_abstained:
        if gen_output.abstained:
            return "refused_correctly"
        return "wrong"

    # Question is answerable (expected_abstained == False)
    if gen_output.abstained:
        return "refused_wrongly"

    if not citation_valid:
        return "wrong"

    # Check that at least one citation references the expected paper and version
    cited_papers = {
        (loc.paper_id.strip(), loc.arxiv_version.strip().lower())
        for c in gen_output.claims
        for loc in c.citations
    }
    if eval_q.expected_paper_id and eval_q.expected_arxiv_version:
        expected_pair = (
            eval_q.expected_paper_id,
            eval_q.expected_arxiv_version.lower(),
        )
        if expected_pair not in cited_papers:
            return "wrong"

    combined_text = (
        gen_output.answer_text
        + " "
        + " ".join(c.claim_text for c in gen_output.claims)
    ).lower()

    for kw in eval_q.required_all_keywords:
        if kw.lower() not in combined_text:
            return "wrong"

    if eval_q.required_any_keywords:
        if not any(kw.lower() in combined_text for kw in eval_q.required_any_keywords):
            return "wrong"

    for f_kw in eval_q.forbidden_keywords:
        if f_kw.lower() in combined_text:
            return "wrong"

    return "correct"