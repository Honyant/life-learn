"""Tests for the SDFT correction pipeline.

Unit tests mock external dependencies (LLM, API) and run without GPU.
Integration test (marked @pytest.mark.gpu) requires GPU + OPENAI_API_KEY.
"""
import json
import pytest
from unittest.mock import MagicMock, patch

from sdft_correction.correction_detector import (
    detect_correction,
    CorrectionResult,
    _parse_detection_output,
    _format_conversation,
)
from sdft_correction.augmenter import generate_prompt_variations, _parse_variations
from sdft_correction.expert_demos import ExpertDemo, generate_expert_demos
from sdft_correction.data_formatter import format_for_sdft


# ============================================================
# Demo scenario constants
# ============================================================

DEMO_USER_QUESTION = (
    "What are the steps to safely deploy a new feature to production?"
)

DEMO_WRONG_RESPONSE = (
    "Here are the steps to deploy a new feature to production:\n"
    "1. Merge your feature branch into main\n"
    "2. Push to the production server\n"
    "3. Run the automated test suite\n"
    "4. Monitor for errors\n"
    "5. If tests fail, roll back"
)

DEMO_CORRECTION = (
    "That's not right. You should NEVER push to production before running tests. "
    "The correct order is:\n"
    "1. Run the automated test suite on your feature branch\n"
    "2. Get code review approval\n"
    "3. Merge into main\n"
    "4. Run integration tests on main\n"
    "5. Push to production\n"
    "6. Monitor for errors"
)

DEMO_CONVERSATION = [
    {"role": "user", "content": DEMO_USER_QUESTION},
    {"role": "assistant", "content": DEMO_WRONG_RESPONSE},
    {"role": "user", "content": DEMO_CORRECTION},
]


# ============================================================
# correction_detector tests
# ============================================================


class TestCorrectionDetectorParsing:
    def test_parse_valid_json_correction(self):
        raw = '{"is_correction": true, "what_was_wrong": "wrong order", "what_should_be": "tests first"}'
        result = _parse_detection_output(raw)
        assert result["is_correction"] is True
        assert result["what_was_wrong"] == "wrong order"
        assert result["what_should_be"] == "tests first"

    def test_parse_valid_json_no_correction(self):
        raw = '{"is_correction": false, "what_was_wrong": "", "what_should_be": ""}'
        result = _parse_detection_output(raw)
        assert result["is_correction"] is False

    def test_parse_json_with_markdown_fences(self):
        raw = '```json\n{"is_correction": true, "what_was_wrong": "bad", "what_should_be": "good"}\n```'
        result = _parse_detection_output(raw)
        assert result["is_correction"] is True

    def test_parse_json_embedded_in_text(self):
        raw = 'Analysis: {"is_correction": true, "what_was_wrong": "x", "what_should_be": "y"} done.'
        result = _parse_detection_output(raw)
        assert result["is_correction"] is True

    def test_parse_garbage_returns_no_correction(self):
        result = _parse_detection_output("This is not valid JSON at all.")
        assert result["is_correction"] is False


class TestCorrectionDetectorWithMock:
    def test_detects_correction(self):
        mock_llm = MagicMock()
        mock_llm.generate.return_value = json.dumps(
            {
                "is_correction": True,
                "what_was_wrong": "Steps in wrong order - pushing before testing",
                "what_should_be": "Run tests before merging and pushing to production",
            }
        )

        result = detect_correction(DEMO_CONVERSATION, mock_llm)
        assert result.is_correction is True
        assert result.original_model_response == DEMO_WRONG_RESPONSE

    def test_no_correction_for_followup(self):
        mock_llm = MagicMock()
        mock_llm.generate.return_value = json.dumps(
            {"is_correction": False, "what_was_wrong": "", "what_should_be": ""}
        )

        conversation = [
            {"role": "user", "content": "How do I deploy?"},
            {"role": "assistant", "content": "Here are the steps..."},
            {"role": "user", "content": "Can you explain step 2 in more detail?"},
        ]
        result = detect_correction(conversation, mock_llm)
        assert result.is_correction is False

    def test_raises_on_empty_conversation(self):
        with pytest.raises(ValueError):
            detect_correction([], MagicMock())

    def test_raises_when_last_message_not_user(self):
        with pytest.raises(ValueError):
            detect_correction(
                [{"role": "assistant", "content": "hello"}], MagicMock()
            )


class TestFormatConversation:
    def test_formats_correctly(self):
        result = _format_conversation(DEMO_CONVERSATION)
        assert "USER:" in result
        assert "ASSISTANT:" in result
        assert DEMO_USER_QUESTION in result


# ============================================================
# augmenter tests
# ============================================================


class TestAugmenterParsing:
    def test_parse_valid_array(self):
        raw = '["Question one?", "Question two?", "Question three?"]'
        result = _parse_variations(raw)
        assert len(result) == 3
        assert all(isinstance(q, str) for q in result)

    def test_parse_with_code_fences(self):
        raw = '```json\n["Q1?", "Q2?"]\n```'
        result = _parse_variations(raw)
        assert len(result) == 2

    def test_parse_garbage_returns_empty(self):
        result = _parse_variations("no json here")
        assert result == []

    def test_filters_non_strings(self):
        raw = '["Valid?", 123, "Also valid?", null]'
        result = _parse_variations(raw)
        assert result == ["Valid?", "Also valid?"]


class TestAugmenterWithMock:
    def test_generates_variations(self):
        mock_llm = MagicMock()
        mock_llm.generate.return_value = json.dumps(
            [
                "How do you prepare for a job interview?",
                "What steps should I follow to submit a research paper?",
            ]
        )

        results = generate_prompt_variations(
            what_was_wrong="Steps in wrong order",
            what_should_be="Run tests before deploying",
            original_question=DEMO_USER_QUESTION,
            original_wrong_response=DEMO_WRONG_RESPONSE,
            llm=mock_llm,
            n=2,
        )
        assert len(results) == 2
        assert all(isinstance(q, str) for q in results)


# ============================================================
# expert_demos tests
# ============================================================


class TestExpertDemos:
    @patch("sdft_correction.expert_demos.OpenAI")
    def test_generates_demos(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Expert answer here."
        mock_client.chat.completions.create.return_value = mock_response

        results = generate_expert_demos(
            prompts=["Q1?", "Q2?"],
            what_was_wrong="wrong order",
            what_should_be="correct order",
            demos_per_prompt=2,
        )
        assert len(results) == 4
        assert all(isinstance(d, ExpertDemo) for d in results)
        assert all(d.demonstration == "Expert answer here." for d in results)

    @patch("sdft_correction.expert_demos.OpenAI")
    def test_handles_api_failure_gracefully(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.side_effect = Exception("API error")

        results = generate_expert_demos(
            prompts=["Q1?"],
            what_was_wrong="x",
            what_should_be="y",
            demos_per_prompt=2,
        )
        assert len(results) == 0


# ============================================================
# data_formatter tests
# ============================================================


class TestDataFormatter:
    def test_format_for_sdft_dataset(self):
        demos = [
            ExpertDemo(prompt="Q1?", demonstration="Correct A1"),
            ExpertDemo(prompt="Q2?", demonstration="Correct A2"),
            ExpertDemo(prompt="Q1?", demonstration="Correct A1 variant"),
        ]
        dataset = format_for_sdft(demos)

        assert len(dataset) == 3
        assert "prompt" in dataset.column_names
        assert "teacher_prompt" in dataset.column_names

    def test_prompt_is_student_input(self):
        demos = [ExpertDemo(prompt="How to deploy?", demonstration="Run tests first.")]
        dataset = format_for_sdft(demos)

        prompt = dataset[0]["prompt"]
        assert isinstance(prompt, list)
        assert prompt[0]["role"] == "user"
        assert prompt[0]["content"] == "How to deploy?"

    def test_teacher_prompt_matches_paper_template(self):
        """Verify teacher prompt matches Section 3 of the paper."""
        demos = [ExpertDemo(prompt="My question?", demonstration="Expert demo text.")]
        dataset = format_for_sdft(demos)

        teacher_content = dataset[0]["teacher_prompt"][0]["content"]

        # Must contain the question
        assert "My question?" in teacher_content
        # Must contain the paper's template phrasing
        assert "This is an example for a response to the question:" in teacher_content
        # Must contain the expert demonstration
        assert "Expert demo text." in teacher_content
        # Must contain the generation instruction
        assert "Now answer with a response of your own, including the thinking process." in teacher_content

    def test_teacher_prompt_does_not_leak_into_student(self):
        demos = [ExpertDemo(prompt="Q?", demonstration="Secret expert answer")]
        dataset = format_for_sdft(demos)

        student_content = dataset[0]["prompt"][0]["content"]
        assert "Secret expert answer" not in student_content
        assert "example for a response" not in student_content

    def test_dataset_format_matches_distiltrainer(self):
        """Verify structure matches what DistilTrainer expects."""
        demos = [
            ExpertDemo(prompt="Q1?", demonstration="A1"),
            ExpertDemo(prompt="Q2?", demonstration="A2"),
        ]
        dataset = format_for_sdft(demos)

        for i in range(len(dataset)):
            prompt = dataset[i]["prompt"]
            teacher = dataset[i]["teacher_prompt"]

            assert isinstance(prompt, list)
            assert isinstance(teacher, list)
            assert all(isinstance(m, dict) for m in prompt)
            assert all(isinstance(m, dict) for m in teacher)
            assert all("role" in m and "content" in m for m in prompt)
            assert all("role" in m and "content" in m for m in teacher)


# ============================================================
# Integration test (requires GPU + OPENAI_API_KEY)
# ============================================================


@pytest.mark.gpu
class TestFullPipelineIntegration:
    """End-to-end test. Run with: pytest -m gpu -s"""

    def test_full_pipeline(self):
        from sdft_correction.inference import LocalInference
        from sdft_correction.config import PipelineConfig
        from sdft_correction.chat import _load_trained_model, _load_env

        config = PipelineConfig()
        _load_env(config)

        # 1. Load model and check baseline
        llm = LocalInference(config.model_name)

        baseline_response = llm.generate(
            [{"role": "user", "content": DEMO_USER_QUESTION}],
            max_new_tokens=512,
            temperature=0.1,
            do_sample=False,
        )
        print(f"\n=== BASELINE RESPONSE ===\n{baseline_response}")

        # 2. Detect correction
        result = detect_correction(DEMO_CONVERSATION, llm)
        assert result.is_correction, (
            f"Failed to detect correction. Raw: {result.raw_llm_output}"
        )
        print(f"\n=== CORRECTION DETECTED ===")
        print(f"  Wrong: {result.what_was_wrong}")
        print(f"  Should be: {result.what_should_be}")

        # 3. Generate prompt variations
        variations = generate_prompt_variations(
            what_was_wrong=result.what_was_wrong,
            what_should_be=result.what_should_be,
            original_question=DEMO_USER_QUESTION,
            original_wrong_response=DEMO_WRONG_RESPONSE,
            llm=llm,
            n=config.num_prompt_variations,
        )
        all_prompts = [DEMO_USER_QUESTION] + variations
        assert len(all_prompts) >= 3, f"Only got {len(all_prompts)} prompts"
        print(f"\n=== PROMPTS ({len(all_prompts)}) ===")
        for i, p in enumerate(all_prompts[:5]):
            print(f"  {i}: {p[:80]}...")

        # 4. Unload local model, generate expert demos
        llm.unload()

        expert_demo_list = generate_expert_demos(
            prompts=all_prompts,
            what_was_wrong=result.what_was_wrong,
            what_should_be=result.what_should_be,
            demos_per_prompt=config.num_expert_demos_per_prompt,
            model=config.openai_model,
        )
        assert len(expert_demo_list) >= 5, (
            f"Only got {len(expert_demo_list)} demos"
        )
        print(f"\n=== EXPERT DEMOS: {len(expert_demo_list)} ===")

        # 5. Format dataset
        dataset = format_for_sdft(expert_demo_list)
        print(f"=== DATASET: {len(dataset)} training pairs ===")

        # 6. Train
        model_path = run_sdft_training(
            dataset=dataset,
            model_name=config.model_name,
            output_dir=str(config.output_dir),
            learning_rate=config.learning_rate,
            num_train_epochs=config.num_train_epochs,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            max_prompt_length=config.max_prompt_length,
            max_completion_length=config.max_completion_length,
        )

        # 7. Verify on analogous question (fresh context)
        trained_llm = _load_trained_model(model_path)

        verification_question = (
            "What are the steps to release a new version of a mobile app?"
        )
        response = trained_llm.generate(
            [{"role": "user", "content": verification_question}],
            max_new_tokens=512,
            temperature=0.1,
            do_sample=False,
        )

        print(f"\n=== VERIFICATION ===")
        print(f"Q: {verification_question}")
        print(f"A: {response}")

        # Check that testing comes before release/deploy/publish
        response_lower = response.lower()
        test_pos = response_lower.find("test")
        release_pos = max(
            response_lower.find("release"),
            response_lower.find("deploy"),
            response_lower.find("publish"),
            response_lower.find("submit"),
        )

        assert test_pos != -1, f"Response doesn't mention testing:\n{response}"
        assert release_pos != -1, f"Response doesn't mention releasing:\n{response}"
        assert test_pos < release_pos, (
            f"Testing (pos {test_pos}) should come before "
            f"release (pos {release_pos}):\n{response}"
        )
