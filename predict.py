import logging
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.preprocessing import MultiLabelBinarizer as MLB

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the classes (ATT&CK techniques) the model is trained on
CLASSES = [
    "T1003.001",
    "T1005",
    "T1012",
    "T1016",
    "T1021.001",
    "T1027",
    "T1033",
    "T1036.005",
    "T1041",
    "T1047",
    "T1053.005",
    "T1055",
    "T1056.001",
    "T1057",
    "T1059.003",
    "T1068",
    "T1070.004",
    "T1071.001",
    "T1072",
    "T1074.001",
    "T1078",
    "T1082",
    "T1083",
    "T1090",
    "T1095",
    "T1105",
    "T1106",
    "T1110",
    "T1112",
    "T1113",
    "T1140",
    "T1190",
    "T1204.002",
    "T1210",
    "T1218.011",
    "T1219",
    "T1484.001",
    "T1518.001",
    "T1543.003",
    "T1547.001",
    "T1548.002",
    "T1552.001",
    "T1557.001",
    "T1562.001",
    "T1564.001",
    "T1566.001",
    "T1569.002",
    "T1570",
    "T1573.001",
    "T1574.002",
]

model = BertForSequenceClassification.from_pretrained(
    "./fine_tuned_scibert_multi_label"
)
tokenizer = BertTokenizer.from_pretrained("./fine_tuned_scibert_multi_label")
mlb = MLB()
mlb.fit([[c] for c in CLASSES])


def predict_techniques(text, threshold=0.5):
    inputs = tokenizer(
        text, return_tensors="pt", padding="max_length", truncation=True, max_length=512
    )
    with torch.no_grad():
        outputs = model(**inputs)

    # Apply sigmoid to get probabilities
    probabilities = torch.sigmoid(outputs.logits)[0].numpy()

    # Get predictions based on threshold
    predictions = probabilities > threshold

    # Convert to technique IDs
    predicted_techniques = mlb.inverse_transform(predictions.reshape(1, -1))[0]

    # Return techniques with their confidence scores
    technique_scores = {
        CLASSES[i]: float(probabilities[i])
        for i in range(len(CLASSES))
        if probabilities[i] > threshold
    }
    return sorted(technique_scores.items(), key=lambda x: x[1], reverse=True)


def predict():
    example_text = "The malware uses PowerShell to execute commands and steal credentials from the system."
    predicted = predict_techniques(example_text, threshold=0.1)
    print("\nSample Prediction:")
    print(f"Text: {example_text}")
    print("Predicted techniques and confidence scores:")
    for technique, score in predicted:
        print(f"  {technique}: {score:.4f}")


if __name__ == "__main__":
    predict()
