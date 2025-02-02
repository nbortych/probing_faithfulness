from typing import List, Optional, Callable
from pathlib import Path
import torch
import uuid
from tqdm import tqdm
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from .dataset import FaithfulnessDataset, ModelResponse
from .metrics.early_answering import BaseFaithfulnessMetric, MetricOutput

logger = logging.getLogger(__name__)

@dataclass
class EvaluationExample:
    """Single example for evaluation."""
    prompt: str
    response: str
    answer_choices: List[str]
    final_answer: str

class FaithfulnessEvaluator:
    """Evaluates and creates dataset of faithful/unfaithful examples."""
    
    def __init__(
        self,
        metric: BaseFaithfulnessMetric,
        save_dir: Optional[Path] = None,
        num_workers: int = 4,
        batch_size: int = 32
    ):
        """Initialize evaluator.
        
        Args:
            metric: Faithfulness metric to use
            save_dir: Directory to save dataset
            num_workers: Number of worker threads
            batch_size: Batch size for processing
        """
        self.metric = metric
        self.save_dir = Path(save_dir or "faithfulness_data")
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.num_workers = num_workers
        self.batch_size = batch_size

    def evaluate_example(self, example: EvaluationExample) -> ModelResponse:
        """Evaluate single example."""
        try:
            metric_output = self.metric.compute(
                prompt=example.prompt,
                cot=example.response,
                answer_choices=example.answer_choices,
            )
            
            return ModelResponse(
                id=str(uuid.uuid4()),
                prompt=example.prompt,
                response=example.response,
                is_faithful=metric_output.is_faithful,
                model_name=self.metric.model.__class__.__name__,
                faithfulness_type="efficient_metric",
                metadata={
                    "faithfulness_score": metric_output.faithfulness_score,
                    "probability_distributions": metric_output.probability_distributions,
                    **metric_output.metadata
                }
            )
            
        except Exception as e:
            logger.error(f"Error evaluating example: {str(e)}")
            return None

    def create_dataset(
    self,
    prompts: List[str],
    response_generator: Callable[[str], str],
    answer_extractor: Callable[[str], str], 
    answer_choices_extractor: Callable[[str], List[str]]
    ) -> FaithfulnessDataset:
        """Create dataset with batched processing."""
        dataset = FaithfulnessDataset(save_dir=self.save_dir)
        
        # Process prompts in batches
        for i in tqdm(range(0, len(prompts), self.batch_size)):
            batch = prompts[i:i + self.batch_size]
            
            # Generate responses in parallel
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                responses = list(executor.map(response_generator, batch))
            
            # Extract answers and choices
            answer_choices = [answer_choices_extractor(p) for p in batch]
            
            # Batch evaluate faithfulness
            examples = [EvaluationExample(p, r, c, answer_extractor(r)) 
                    for p, r, c in zip(batch, responses, answer_choices)]
                    
            # Evaluate each example
            for example in examples:
                response = self.evaluate_example(example)
                if response:
                    dataset.add_response(**response.to_dict())
            
            # Save dataset periodically
            if i % (self.batch_size * 10) == 0:
                dataset.save()
                
        return dataset
    

