import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import json
import pickle
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        Q = self.w_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attention = F.softmax(scores, dim=-1)
        context = torch.matmul(attention, V)
        
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model)
        
        return self.w_o(context)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        attn_output = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

class IntentClassifier(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_heads=8, n_layers=6, 
                 n_classes=20, max_len=128, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_model * 4, dropout)
            for _ in range(n_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_classes)
        )
        
    def forward(self, x, mask=None):
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        
        for transformer in self.transformer_blocks:
            x = transformer(x, mask)
            
        x = self.norm(x)
        x = x.mean(dim=1)
        
        return self.classifier(x)

class TextProcessor:
    def __init__(self):
        self.vocab = {}
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.max_len = 128
        
    def build_vocab(self, texts):
        word_freq = {}
        for text in texts:
            tokens = self.tokenize(text)
            for token in tokens:
                word_freq[token] = word_freq.get(token, 0) + 1
        
        vocab_list = ['<PAD>', '<UNK>'] + [word for word, freq in 
                     sorted(word_freq.items(), key=lambda x: x[1], reverse=True) 
                     if freq >= 2]
        
        self.word_to_idx = {word: idx for idx, word in enumerate(vocab_list)}
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
        self.vocab_size = len(vocab_list)
        
    def tokenize(self, text):
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())
        tokens = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        return [token for token in tokens if token not in stop_words and len(token) > 1]
    
    def encode(self, text):
        tokens = self.tokenize(text)
        indices = [self.word_to_idx.get(token, self.word_to_idx['<UNK>']) 
                  for token in tokens]
        
        if len(indices) > self.max_len:
            indices = indices[:self.max_len]
        else:
            indices += [self.word_to_idx['<PAD>']] * (self.max_len - len(indices))
            
        return torch.tensor(indices, dtype=torch.long)

class AcademicDataset(Dataset):
    def __init__(self, texts, labels, processor):
        self.texts = texts
        self.labels = labels
        self.processor = processor
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoded_text = self.processor.encode(text)
        return encoded_text, torch.tensor(label, dtype=torch.long)

def create_sample_data():
    intents = [
        "course_info", "assignment_help", "exam_schedule", "grades_inquiry",
        "registration", "financial_aid", "library_hours", "campus_events",
        "technical_support", "career_services", "housing_info", "dining_hours",
        "parking_info", "health_services", "counseling", "academic_calendar",
        "professor_contact", "tutoring", "study_groups", "graduation_requirements"
    ]
    
    sample_queries = {
        "course_info": [
            "What are the prerequisites for CS101?",
            "When is the next offering of Linear Algebra?",
            "Can you tell me about the Data Structures course?",
            "What textbooks are required for Physics 201?",
            "How many credits is the Biology lab?"
        ],
        "assignment_help": [
            "I need help with my calculus homework",
            "Can you explain the requirements for the research paper?",
            "What is the due date for the programming assignment?",
            "How do I submit my lab report?",
            "I'm struggling with the chemistry problem set"
        ],
        "exam_schedule": [
            "When is the final exam for Statistics?",
            "What time is the midterm for History 101?",
            "Can you show me the exam schedule?",
            "Where is the Chemistry final being held?",
            "How long is the English literature exam?"
        ],
        "grades_inquiry": [
            "What is my current GPA?",
            "Can you check my grade for the last assignment?",
            "Why haven't my test scores been posted?",
            "How do I view my transcript?",
            "When will final grades be available?"
        ],
        "registration": [
            "How do I register for next semester?",
            "Can I add another course to my schedule?",
            "What is the registration deadline?",
            "I need to drop a class, how do I do that?",
            "Is there a waitlist for Biology 301?"
        ]
    }
    
    texts = []
    labels = []
    label_encoder = LabelEncoder()
    
    for intent, queries in sample_queries.items():
        for query in queries:
            texts.append(query)
            labels.append(intent)
    
    encoded_labels = label_encoder.fit_transform(labels)
    return texts, encoded_labels.tolist(), label_encoder

def train_model():
    texts, labels, label_encoder = create_sample_data()
    
    processor = TextProcessor()
    processor.build_vocab(texts)
    
    dataset = AcademicDataset(texts, labels, processor)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    model = IntentClassifier(
        vocab_size=processor.vocab_size,
        n_classes=len(label_encoder.classes_)
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(100):
        total_loss = 0
        for batch_texts, batch_labels in dataloader:
            optimizer.zero_grad()
            
            outputs = model(batch_texts)
            loss = criterion(outputs, batch_labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        if epoch % 20 == 0:
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'processor': processor,
        'label_encoder': label_encoder,
        'model_config': {
            'vocab_size': processor.vocab_size,
            'n_classes': len(label_encoder.classes_)
        }
    }, 'academic_chatbot_model.pth')
    
    return model, processor, label_encoder

class ChatbotInference:
    def __init__(self, model_path):
        checkpoint = torch.load(model_path, map_location='cpu')
        
        self.processor = checkpoint['processor']
        self.label_encoder = checkpoint['label_encoder']
        config = checkpoint['model_config']
        
        self.model = IntentClassifier(
            vocab_size=config['vocab_size'],
            n_classes=config['n_classes']
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        self.responses = {
            "course_info": "I can help you find course information. Please specify which course you're interested in.",
            "assignment_help": "I'm here to assist with your assignments. What specific help do you need?",
            "exam_schedule": "Let me help you with exam scheduling information. Which course exam are you asking about?",
            "grades_inquiry": "I can help you with grade-related questions. Please check your student portal for detailed grades.",
            "registration": "For registration assistance, please visit the registrar's office or use the online portal."
        }
    
    def predict_intent(self, text):
        with torch.no_grad():
            encoded_text = self.processor.encode(text).unsqueeze(0)
            output = self.model(encoded_text)
            predicted_class = torch.argmax(output, dim=1).item()
            confidence = torch.softmax(output, dim=1).max().item()
            
            intent = self.label_encoder.inverse_transform([predicted_class])[0]
            return intent, confidence
    
    def generate_response(self, text):
        intent, confidence = self.predict_intent(text)
        
        if confidence > 0.7:
            response = self.responses.get(intent, "I'm not sure how to help with that. Please contact student services.")
        else:
            response = "I'm not sure I understand. Could you please rephrase your question?"
            
        return {
            "response": response,
            "intent": intent,
            "confidence": confidence
        }

if __name__ == "__main__":
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    
    print("Training model...")
    model, processor, label_encoder = train_model()
    print("Model training completed!")
    
    chatbot = ChatbotInference('academic_chatbot_model.pth')
    
    test_queries = [
        "What are the prerequisites for the computer science course?",
        "I need help with my math assignment",
        "When is the final exam for biology?",
        "What is my current grade in physics?"
    ]
    
    for query in test_queries:
        result = chatbot.generate_response(query)
        print(f"Query: {query}")
        print(f"Intent: {result['intent']} (Confidence: {result['confidence']:.3f})")
        print(f"Response: {result['response']}\n")