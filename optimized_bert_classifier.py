import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel
import numpy as np
import json
import pickle
import time
from torch.quantization import quantize_dynamic

class OptimizedBertClassifier(nn.Module):
    def __init__(self, num_intents, hidden_size=384, dropout_rate=0.2):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Sequential(
            nn.Linear(768, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, num_intents)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        output = self.dropout(pooled_output)
        return self.classifier(output)

class KnowledgeDistillationTrainer:
    def __init__(self, teacher_model, student_model, tokenizer, intent_labels):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.teacher = teacher_model.to(self.device)
        self.student = student_model.to(self.device)
        self.tokenizer = tokenizer
        self.intent_labels = intent_labels
        self.teacher.eval()
        self.student.train()

    def distillation_loss(self, student_logits, teacher_logits, labels, temperature=2.0, alpha=0.7):
        soft_loss = nn.KLDivLoss(reduction='batchmean')(
            F.log_softmax(student_logits / temperature, dim=-1),
            F.softmax(teacher_logits / temperature, dim=-1)
        ) * (temperature ** 2)
        hard_loss = F.cross_entropy(student_logits, labels)
        return alpha * soft_loss + (1.0 - alpha) * hard_loss

    def train(self, train_data, num_epochs=5, batch_size=32, learning_rate=3e-5):
        optimizer = torch.optim.Adam(self.student.parameters(), lr=learning_rate)
        train_size = len(train_data['input_ids'])

        for epoch in range(num_epochs):
            total_loss = 0
            for i in range(0, train_size, batch_size):
                batch_end = min(i + batch_size, train_size)
                
                input_ids = train_data['input_ids'][i:batch_end].to(self.device)
                attention_masks = train_data['attention_masks'][i:batch_end].to(self.device)
                labels = train_data['labels'][i:batch_end].to(self.device)

                optimizer.zero_grad()

                with torch.no_grad():
                    teacher_logits = self.teacher(input_ids, attention_masks)

                student_logits = self.student(input_ids, attention_masks)
                loss = self.distillation_loss(student_logits, teacher_logits, labels)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / (train_size // batch_size)
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}')

        return self.student

class OptimizedIntentPredictor:
    def __init__(self, model_path, tokenizer_path, intent_labels_path):
        self.device = torch.device('cpu')
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
        
        with open(intent_labels_path, 'r') as f:
            self.intent_labels = json.load(f)

        self.model = OptimizedBertClassifier(len(self.intent_labels))
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model = quantize_dynamic(self.model, {nn.Linear}, dtype=torch.qint8)
        self.model.eval()

    def predict(self, text, confidence_threshold=0.75):
        start_time = time.time()
        
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            padding=True,
            max_length=128
        )

        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)
            probabilities = F.softmax(outputs, dim=-1)
            confidence, predicted_idx = torch.max(probabilities, dim=-1)

        predicted_intent = self.intent_labels[predicted_idx.item()]
        confidence_score = confidence.item()

        if confidence_score < confidence_threshold:
            predicted_intent = 'unknown'

        inference_time = (time.time() - start_time) * 1000

        return {
            'intent': predicted_intent,
            'confidence': confidence_score,
            'inference_time_ms': inference_time
        }

def prepare_training_data(data_path, tokenizer, max_length=128):
    with open(data_path, 'r') as f:
        raw_data = json.load(f)

    intent_labels = {item['intent']: idx for idx, item in enumerate(sorted(set(item['intent'] for item in raw_data)))}
    
    input_ids = []
    attention_masks = []
    labels = []

    for item in raw_data:
        encoded = tokenizer(
            item['text'],
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )
        
        input_ids.append(encoded['input_ids'])
        attention_masks.append(encoded['attention_mask'])
        labels.append(intent_labels[item['intent']])

    return {
        'input_ids': torch.cat(input_ids, dim=0),
        'attention_masks': torch.cat(attention_masks, dim=0),
        'labels': torch.tensor(labels),
        'intent_labels': intent_labels
    }

def optimize_model(data_path, output_dir):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    train_data = prepare_training_data(data_path, tokenizer)

    teacher_model = IntentClassifier(len(train_data['intent_labels']))
    student_model = OptimizedBertClassifier(len(train_data['intent_labels']))

    trainer = KnowledgeDistillationTrainer(teacher_model, student_model, tokenizer, train_data['intent_labels'])
    optimized_model = trainer.train(train_data)

    os.makedirs(output_dir, exist_ok=True)
    
    model_path = os.path.join(output_dir, 'optimized_model.pth')
    tokenizer_path = os.path.join(output_dir, 'tokenizer')
    labels_path = os.path.join(output_dir, 'intent_labels.json')

    torch.save(optimized_model.state_dict(), model_path)
    tokenizer.save_pretrained(tokenizer_path)
    with open(labels_path, 'w') as f:
        json.dump(train_data['intent_labels'], f)

    predictor = OptimizedIntentPredictor(model_path, tokenizer_path, labels_path)
    return predictor

if __name__ == "__main__":
    output_dir = "optimized_bert_model"
    data_path = "training_data.json"
    predictor = optimize_model(data_path, output_dir)
    
    test_queries = [
        "What are the course requirements for CS101?",
        "When is the final exam scheduled?",
        "I need help with my math homework"
    ]
    
    for query in test_queries:
        result = predictor.predict(query)
        print(f"Query: {query}")
        print(f"Intent: {result['intent']}")
        print(f"Confidence: {result['confidence']:.3f}")
        print(f"Inference Time: {result['inference_time_ms']:.2f}ms\n")