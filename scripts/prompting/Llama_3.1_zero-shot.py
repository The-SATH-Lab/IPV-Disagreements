import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, balanced_accuracy_score, f1_score, recall_score

# Constants
BASE_MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# Load model and tokenizer
def load_model_and_tokenizer(base_model_name):
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map="auto"
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer

model, tokenizer = load_model_and_tokenizer(BASE_MODEL_NAME)

# Load and preprocess test dataset
def load_and_prepare_data(file_path):
    df = pd.read_json(file_path)
    df.rename(columns={'Conversational History': 'text'}, inplace=True)
    return df

test_df = load_and_prepare_data("snap_test_messages.json")

# Prompt generation for zero-shot inference
def generate_test_prompt(text):
    return f"""
    Analyze the conversation between romantic partners enclosed in square brackets. 
    Determine if there is a disagreement, and output either "Yes" or "No" as the label. 
    Output your prediction next to "label:".
    Do not return (or) add anything else. 

    [text: {text}]
    label:
    """.strip()

# Prediction function
def predict(X_test, pipe, max_retries=5):
    y_pred = []
    categories = ["Yes", "No"]
    
    for i in tqdm(range(len(X_test))):
        retry_count = 0
        while retry_count < max_retries:
            try:
                prompt = X_test.iloc[i]["text"]
                result = pipe(prompt)
                answer = result[0]['generated_text'].split("label:")[-1].strip()
                print(f"{i+1}: {answer}")
                
                for category in categories:
                    if category.lower() in answer.lower():
                        y_pred.append(category)
                        break
                else:
                    retry_count += 1
                    continue
                
                break
            except Exception as e:
                print(f"Error occurred: {e}. Retrying...")
                retry_count += 1
        
        if retry_count == max_retries:
            y_pred.append("none")
    
    return y_pred

# Evaluation function
def evaluate(y_true, y_pred):
    labels = ['Yes', 'No']
    mapping = {'Yes': 1, 'No': 0}
    
    y_true_mapped = np.vectorize(mapping.get)(y_true)
    y_pred_mapped = np.vectorize(mapping.get)(y_pred)
    
    accuracy = accuracy_score(y_true=y_true_mapped, y_pred=y_pred_mapped)
    print(f'Accuracy: {accuracy:.3f}')
    
    print("\nClassification Report:")
    print(classification_report(y_true=y_true_mapped, y_pred=y_pred_mapped, target_names=labels))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true=y_true_mapped, y_pred=y_pred_mapped))

# Main pipeline for model evaluation
def fit_and_return_results(X_test, y_true, model, tokenizer, generation_configs):
    balanced_accuracies, f1_macro_scores, recall_1_scores = [], [], []
    classification_reports = []

    for config_name, gen_config in generation_configs.items():
        print(f"\nUsing generation config: {config_name}")
        
        pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, **gen_config)
        y_pred = predict(X_test, pipe)
        evaluate(y_true, y_pred)
        
        balanced_accuracies.append(balanced_accuracy_score(y_true, y_pred))
        f1_macro_scores.append(f1_score(y_true, y_pred, average='macro'))
        recall_1_scores.append(recall_score(y_true, y_pred, pos_label="Yes"))
        
        classification_reports.append(classification_report(y_true, y_pred, output_dict=True))

    # Summary of results
    avg_balanced_accuracy = np.mean(balanced_accuracies)
    avg_f1_macro_score = np.mean(f1_macro_scores)
    avg_recall_1_score = np.mean(recall_1_scores)

    print(f"\nAverage Balanced Accuracy: {avg_balanced_accuracy}")
    print(f"Average F-1 Score (Macro): {avg_f1_macro_score}")
    print(f"Average Recall Score (Class 1): {avg_recall_1_score}")

    return balanced_accuracies, f1_macro_scores, recall_1_scores

# Generation configurations
generation_configs = {
    "Greedy Search": {
        "num_beams": 1,
        "do_sample": False,
        "max_new_tokens": 1,
        "early_stopping": True
    },
    "Beam Search": {
        "num_beams": 2,
        "do_sample": False,
        "max_new_tokens": 1,
        "early_stopping": True
    },
    "Sampling Top-k": {
        "do_sample": True,
        "num_beams": 1,
        "max_new_tokens": 1,
        "early_stopping": True,
        "temperature": 0.2,
        "top_k": 50
    },
    "Sampling Top-p": {
        "do_sample": True,
        "top_k": 0,
        "num_beams": 1,
        "max_new_tokens": 1,
        "early_stopping": True,
        "temperature": 0.3,
        "top_p": 0.9
    }
}

# Load data
train_df = load_and_prepare_data("snap_train_messages.json")
test_df = load_and_prepare_data("snap_test_messages.json")

# Train-test split
train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42, stratify=train_df['Disagreement'])

# Apply prompt generation
few_shot_examples = json.dumps([], indent=4)  # Provide your few-shot examples
X_test = test_df['text'].apply(lambda text: generate_test_prompt(text, few_shot_examples))

y_true = test_df['Disagreement']

# Load model and tokenizer
model_name = "mistralai/Mistral-7B-Instruct-v0.3"
model, tokenizer = load_model_and_tokenizer(model_name)

# Train and evaluate
key_metric_results = fit_and_return_results(X_test, y_true, model, tokenizer, generation_configs)

# Evaluate on social test set
social_df = load_and_prepare_data("social_messages.json")
X_test_social = social_df['text'].apply(lambda text: generate_test_prompt(text, few_shot_examples))
y_true_social = social_df['Disagreement']

key_metric_results_social = fit_and_return_results(X_test_social, y_true_social, model, tokenizer, generation_configs)
