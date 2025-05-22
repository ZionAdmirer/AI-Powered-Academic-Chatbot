import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel
import numpy as np
import json
import pickle

class IntentClassifier(nn.Module):
    def __init__(self, num_intents, dropout_rate=0.3):
        super(IntentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(768, num_intents)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        output = self.dropout(pooled_output)
        return self.classifier(output)

class IntentPredictor:
    def __init__(self, model_path, tokenizer_path, intent_labels_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
        
        with open(intent_labels_path, 'r') as f:
            self.intent_labels = json.load(f)
        
        self.model = IntentClassifier(len(self.intent_labels))
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
    def predict(self, text, confidence_threshold=0.7):
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
        
        return {
            'intent': predicted_intent,
            'confidence': confidence_score,
            'all_probabilities': probabilities.cpu().numpy().tolist()[0]
        }

def preprocess_text(text):
    import re
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    return ' '.join(tokens)

class DatasetProcessor:
    def __init__(self):
        self.intent_labels = {}
        self.processed_data = []
        
    def load_training_data(self, data_path):
        with open(data_path, 'r') as f:
            raw_data = json.load(f)
        
        unique_intents = set()
        for item in raw_data:
            unique_intents.add(item['intent'])
        
        self.intent_labels = {intent: idx for idx, intent in enumerate(sorted(unique_intents))}
        
        for item in raw_data:
            processed_text = preprocess_text(item['text'])
            self.processed_data.append({
                'text': processed_text,
                'intent': item['intent'],
                'intent_id': self.intent_labels[item['intent']]
            })
        
        return self.processed_data, self.intent_labels
    
    def create_training_dataset(self, tokenizer, max_length=128):
        input_ids = []
        attention_masks = []
        labels = []
        
        for item in self.processed_data:
            encoded = tokenizer(
                item['text'],
                truncation=True,
                padding='max_length',
                max_length=max_length,
                return_tensors='pt'
            )
            
            input_ids.append(encoded['input_ids'])
            attention_masks.append(encoded['attention_mask'])
            labels.append(item['intent_id'])
        
        return {
            'input_ids': torch.cat(input_ids, dim=0),
            'attention_masks': torch.cat(attention_masks, dim=0),
            'labels': torch.tensor(labels)
        }

def train_model(train_dataset, val_dataset, num_epochs=10, learning_rate=2e-5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = IntentClassifier(len(train_dataset['labels'].unique()))
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    train_size = len(train_dataset['labels'])
    val_size = len(val_dataset['labels'])
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for i in range(0, train_size, 32):
            batch_end = min(i + 32, train_size)
            
            input_ids = train_dataset['input_ids'][i:batch_end].to(device)
            attention_masks = train_dataset['attention_masks'][i:batch_end].to(device)
            labels = train_dataset['labels'][i:batch_end].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_masks)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for i in range(0, val_size, 32):
                batch_end = min(i + 32, val_size)
                
                input_ids = val_dataset['input_ids'][i:batch_end].to(device)
                attention_masks = val_dataset['attention_masks'][i:batch_end].to(device)
                labels = val_dataset['labels'][i:batch_end].to(device)
                
                outputs = model(input_ids, attention_masks)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        accuracy = 100 * val_correct / val_total
        avg_loss = total_loss / (train_size // 32)
        
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')
    
    return model