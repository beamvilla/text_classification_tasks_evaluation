from dotenv import load_dotenv
load_dotenv()

import sys
sys.path.append("./")

from src.repositories import TypesafeRepository
from src.models.typesafe_input import QuestionConfig, JevResultType
from src.models.typesafe_output import TypesafeOutput, Answer
from src.models.usage_logs import TokensUsage


typesafe_repository = TypesafeRepository()

response = typesafe_repository.send_request(
    ticket="Our API integration started returning 500 errors on every request about 20 minutes ago, and we can't process any customer orders until this is fixed.",
    question_configs=[
        QuestionConfig(
            title="department",
            result_type=JevResultType.CHOICE,
            instructions="Which team should handle this",
            criteria={
                "billing": "Payment or subscription issues",
                "technical": "Bugs or integration problems",
                "sales": "Pricing or account questions",
            }
        ),
        QuestionConfig(
            title="frustration",
            result_type=JevResultType.SCORE,
            instructions="How frustrated the customer appears",
            criteria=[
                "Calm, just stating facts",
                "Frustrated but civil",
                "Very angry, strong language",
            ]
        ),
        QuestionConfig(
            title="is_urgent",
            result_type=JevResultType.NOUL,
            instructions="The message conveys urgency or time-sensitivity"
        )
    ]
)

token_usage = TokensUsage(
    input_tokens=response.usage.input_tokens,
    output_tokens=response.usage.output_tokens
)

answer = TypesafeOutput.model_validate(response, from_attributes=True)

print(token_usage)
print(answer)