from rasa.nlu.featurizers.featurizer import Featurizer
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.engine.graph import ExecutionContext, GraphComponent
from rasa.nlu.tokenizers.tokenizer import Token, Tokenizer
from rasa.shared.nlu.training_data.features import Features
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.shared.nlu.constants import TEXT, ACTION_TEXT, FEATURE_TYPE_SENTENCE
import tensorflow_hub as hub
import tensorflow as tf
from typing import List
import numpy as np


@DefaultV1Recipe.register(
    DefaultV1Recipe.ComponentType.MESSAGE_FEATURIZER, is_trainable=False
)
class UniversalSentenceEncoderFeaturizer(Featurizer[np.ndarray], GraphComponent):

    @classmethod
    def create(
        cls,
        config,
        model_storage,
        resource,
        execution_context: ExecutionContext,
    ) -> GraphComponent:
        return cls(config,
                   execution_context)

    @classmethod
    def validate_config(cls, config) -> None:
        """Validates that the component is configured properly."""
        pass

    def __init__(
        self, config, execution_context: ExecutionContext
    ) -> None:
        """Initializes the featurizer with the model in the config."""
        super().__init__(execution_context.node_name, config)
        # Load the TensorFlow Hub Module with pre-trained weights

    # URL of the TensorFlow Hub Module
        TFHUB_URL = "https://tfhub.dev/google/universal-sentence-encoder/4"
        self.model = hub.load(TFHUB_URL)

    def process_training_data(self, training_data: TrainingData) -> TrainingData:
        """Processes the training examples in the given training data in-place.
        Args:
          training_data: the training data
        Returns:
          same training data after processing
        """
        self.process(training_data.training_examples)
        return training_data

    def process(self, messages: List[Message]) -> List[Message]:
        for message in messages:
            for attribute in {TEXT, ACTION_TEXT}:
                text = message.get(attribute)
                if text is not None:
                    feature_vector = np.array(self.model([text])[0])
                    feature = Features(
                        feature_vector, FEATURE_TYPE_SENTENCE, attribute, "Universal sentence encoder")
                    message.add_features(feature)
        return messages
