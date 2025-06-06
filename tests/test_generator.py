import pytest
from generation.generator import Generator, GenerationConfig, AnswerGenerator
from unittest.mock import Mock, patch

class MockLLM:
    def generate(self, prompt, **kwargs):
        return "This is a test response."

@pytest.fixture
def mock_llm():
    return MockLLM()

@pytest.fixture
def generator(mock_llm):
    return Generator(mock_llm)

@pytest.fixture
def answer_generator(mock_llm):
    return AnswerGenerator(mock_llm)

def test_generator_initialization():
    mock_llm = MockLLM()
    generator = Generator(mock_llm)
    assert generator.llm == mock_llm
    assert isinstance(generator.config, GenerationConfig)

def test_generator_initialization_without_llm():
    with pytest.raises(ValueError):
        Generator(None)

def test_generate_basic(generator):
    prompt = "Test prompt"
    response = generator.generate(prompt)
    assert response == "This is a test response."

def test_generate_with_parameters(generator):
    prompt = "Test prompt"
    response = generator.generate(
        prompt,
        temperature=0.5,
        max_tokens=100,
        top_p=0.9
    )
    assert response == "This is a test response."

def test_generate_with_empty_prompt(generator):
    with pytest.raises(ValueError):
        generator.generate("")

def test_answer_generator_generate_answer(answer_generator):
    prompt = "Test question"
    answer = answer_generator.generate_answer(prompt)
    assert answer == "This is a test response."

def test_answer_generator_refine_answer(answer_generator):
    raw_answer = "Initial answer"
    context = "Context information"
    refined = answer_generator.refine_answer(raw_answer, context)
    assert refined == "This is a test response."

def test_answer_generator_condense_answer(answer_generator):
    refined_answer = "Detailed answer"
    condensed = answer_generator.condense_answer(refined_answer)
    assert condensed == "This is a test response."

def test_generate_with_retry(generator):
    prompt = "Test prompt"
    response = generator.generate_with_retry(prompt)
    assert response == "This is a test response." 